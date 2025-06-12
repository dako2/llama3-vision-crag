# llm_logger.py

import threading
import json
import os
from typing import Optional

class LLMLogger:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, log_path="all_llm_calls.jsonl"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.log_path = log_path
                cls._instance._buffer = []
            return cls._instance

    def log(self, entry, **kwargs):
        
        entry.update(kwargs)
        self._buffer.append(entry)
        self.flush()

    def flush(self):
        """Write all buffered logs to disk and clear the buffer."""
        if not self._buffer:
            return
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as fout:
            for entry in self._buffer:
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._buffer.clear()

# Create a global instance for convenience
llm_logger = LLMLogger()
