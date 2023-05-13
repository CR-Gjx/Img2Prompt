#!/bin/bash
## QA with the generated caption

#CUDA_VISIBLE_DEVICES='0,1,2,3'
python Imape2Prompt_evaluation.py \
--config ./configs/VQA_test_caption.yaml --dataset 'vqatest' \
--output_dir './output_new/vqa_result/' \
--caption_file '../caption_question_files/vqa_test_caption.json' \
--question_file '../caption_question_files/vqa_test_question.json'  \
--ans_dict_file '../caption_question_files/vqa_test_ans_to_cap_dict.json' \
--min_answer_length 1 --max_answer_length 10 \
--dist_selection 'hugging' \
--batch_size_test 1 --evaluate  --num_caps_per_img 30 --num_question_per_img 30 \
--model_selection 'opt-175b'


