# llama3-vision-crag

Fine tune pipeline 

1. sft_data_gen.py ==> inference and generate sft data ==> validation_sft_response_data_case_5b_web_search_rephrase_w_image.jsonl

2. caption_response.ipynb (case-by-case adjustments for fintuning) ==> validation_sft_response_data_case_5b_web_search_rephrase_w_image.jsonl
 => selected_pipeline_finetune_data_final.jsonl

3. postprocess_data_for_sft.py ==> selected_pipeline_finetune_data_final.jsonl ==> sft_data.jsonl

4. sft_unsloth_v1.py ==> sft_data.jsonl ==> train_conv.pkl

5. merge_model_to_vlllm.py ==> convert and merge  to vllm model


Yi's code ==> evaluate

