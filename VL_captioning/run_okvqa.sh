#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1' python Imape2Prompt_evaluation.py \
--config ./configs/OKVQA_caption.yaml --dataset 'okvqa' \
--output_dir './output_new/vqa_result/' \
--caption_file '../caption_question_files/okvqa_caption.json'  \
--question_file '../caption_question_files/okvqa_question.json' \
--ans_dict_file '../caption_question_files/okvqa_ans_to_cap_dict.json' \
--min_answer_length 1 --max_answer_length 10 \
--dist_selection 'hugging' \
--batch_size_test 1 --evaluate   \
--model_selection 'opt-30b'

