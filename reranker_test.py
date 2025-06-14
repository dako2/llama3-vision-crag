from typing import List, Tuple, Sequence, Union
import os, spacy
from sentence_transformers.cross_encoder import CrossEncoder

import time

import os
import time
from typing import List, Tuple, Sequence, Union

import spacy
from sentence_transformers.cross_encoder import CrossEncoder


class SentenceReranker:
    """
    Sentence reranker with pluggable spaCy model and timing logs.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        force_cpu: bool = True,
        spacy_model: str = "en_core_web_sm",
        disable_pipes: Tuple[str, ...] = ("ner", "tagger", "lemmatizer"),
    ):
        start = time.perf_counter()

        if force_cpu:
            self.device = "cpu"

        print(f"[Init] Loading spaCy model: {spacy_model}")
        if spacy_model == "sentencizer":
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            self.nlp = nlp
        else:
            self.nlp = spacy.load(spacy_model)
            self.nlp.disable_pipes(*disable_pipes)

        print(f"[Init] Loading CrossEncoder: {model_name}")
        self.model = CrossEncoder(model_name, device=self.device)
        self.batch_size = batch_size

        print(f"[Init] Done. Total init time: {time.perf_counter() - start:.2f}s")

    # ---------------- internal helpers -------------------------------------
    def _split(self, text: str) -> List[str]:
        start = time.perf_counter()
        sents = [s.text.strip() for s in self.nlp(text).sents]
        print(f"[Split] Found {len(sents)} sentences in {time.perf_counter() - start:.2f}s")
        return sents

    def _batched_scores(self, pairs: List[Tuple[str, str]]) -> List[float]:
        start = time.perf_counter()
        scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            scores.extend(self.model.predict(batch))
        print(f"[Score] Scored {len(pairs)} pairs in {time.perf_counter() - start:.2f}s")
        return scores

    # ---------------- public interface -------------------------------------
    def rerank(
        self, text: str, query: str, k: int = 3, min_score: float = 0.0
    ) -> List[Tuple[str, float]]:
        sents = self._split(text)
        pairs = [(query, s) for s in sents]
        scores = self._batched_scores(pairs)
        # ranked = [(s, sc) for s, sc in zip(sents, scores) if sc > min_score]
        ranked = [s for s, sc in zip(sents, scores) if sc > min_score]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def batch_rerank(
        self,
        texts: Sequence[str],
        queries: Union[str, Sequence[str]],
        k: int = 3,
        min_score: float = 0.0,
    ) -> List[List[Tuple[str, float]]]:
        if isinstance(queries, str):
            queries = [queries] * len(texts)
        if len(texts) != len(queries):
            raise ValueError("texts and queries length mismatch")
        return [
            self.rerank(t, q, k=k, min_score=min_score)
            for t, q in zip(texts, queries)
        ]


# ---------------- demo ------------------------------------------------------
if __name__ == "__main__":

    txt = """Fr on the bus today it was loud ... quiet video and it said it only went up to around 75 decibels.. maybe I’m just sensitive af ... Full volume will damage hearing over time....Anything over 50% can cause hearing damage. You'd think nobody would make q dangerous product but for some reason they do. With airpods pro, noise canceling can protect your hearing by allowing you to hear the music at 50% volume or less outside of a building in public places, or while driving etc. The exception: if you set reduce loud sounds to 75dB, then any AirPods can be considered safe at any volume. ... This is so incorrect with airpods. 90% volume is 80db. It states on the internet that Airpods can produce 100 decibels, which means playing them at full volume can ... \"With extended exposure, noises that reach a decibel level of 85 can cause permanent damage to the hair cells in the inner ear, leading to hearing loss. Fr on the bus today it was loud as hell so I had to turn my volume on my AirPods to around 80% for just 60 seconds to hear this one video clip and my ears tingling for the next 2 hours ... Edit: I just used the decibel measuring thing that pops up whenever you have AirPods in and it was a pretty quiet video and it said it only went up to around 75 decibels.. maybe I’m just sensitive af ... Full volume will damage hearing over time. seventeen year old here. i’ve been consistently listening to music at an 80-90 decibels range for quite a prolonged period of time; i’d say since at least the ripe ages of eight or nine years old, so for eight or nine years long. frankly, it’s difficult to measure how much hearing damage i’ve sustained, if any at all.\n\n[Info 2] But there&#x27;s been a growing buzz about their impact on ears. Cranking them up may be doing more harm than people realize, experts say. “The World Health Organization right now estimates there&#x27;s 1.5 billion people who live with hearing loss and 430 million of those have a disabling loss.Hearing loss is also possible after just 15 minutes of exposure to an approaching subway train or a car horn at 100 decibels, and standing next to a siren at 120 decibels can cause immediate pain or injury. So regularly exposing your ears to such volumes should raise alarm bells. Did you know that AirPods and Apple Watches come equipped with safety features? AirPods can pump out 100 decibels. That's loud enough to rival the noise of a motorcycle. Headphones and earbuds have become ubiquitous since Apple’s AirPods hit the shelves in 2016. They’re nearly as essential as smartphones. But there's been a growing buzz about their impact on ears. Cranking them up may be doing more harm than people realize, experts say. “The World Health Organization right now estimates there's 1.5 billion people who live with hearing loss and 430 million of those have a disabling loss. It is expected that by 2030, 2.5 billion people will have hearing loss — that's a billion more than right now in six years,\" said Yonah Orlofsky, director of audiology at the New Jersey Eye and Ear medical practice \"By 2050, 700 million people will have disabling hearing loss. “It's understood that at least partly, that number is going up is because of noise exposure via headphone use,” said Orlofsky, whose audiology group has offices in Clifton and Englewood that see 70 patients a day. AirPods can pump out 100 decibels, at their maximum setting. Lahita, who is writing a book that looks at environmental influences on genes, including those that affect hearing, recommends the 60/60 rule, which advises listeners not to exceed 60% of the device’s volume and not to listen for more than 60 minutes a day. Orlofsky agrees and adds that although there’s not a lot of data to prove it, using over-the-ear headphones instead of earbuds may be better. \"Over-the-ear headphones may be a little bit safer than in the ear,\" he said. Since AirPods and similarly designed products sit inside the ear canal, \"sound is actually closer to the eardrum, which is closer to the inner ear, and that may be more impactful than just listening to it over the ear."""
    q = "would wearing this constantly at a 90 decibels volume damage a person's hearing?Airpods"

    texts = ['Full volume will damage hearing over time.', 'It states on the internet that Airpods can produce 100 decibels, which means playing them at full volume can ... "With extended exposure, noises that reach a decibel level of 85 can cause permanent damage to the hair cells in the inner ear, leading to hearing loss.', 'With airpods pro, noise canceling can protect your hearing by allowing you to hear the music at 50% volume or less outside of a building in public places, or while driving etc.', 'AirPods can pump out 100 decibels.', 'AirPods can pump out 100 decibels, at their maximum setting.']
    # Fast rule-based splitter
    fast_rr = SentenceReranker()
    #print("Fast:", fast_rr.rerank(txt, q, k=5, min_score=0))
    results = fast_rr.batch_rerank(texts, q, k=3, min_score=0)
    flat_sentences = [s for group in results for s in group]
    print("Fast:", " ".join(flat_sentences))


