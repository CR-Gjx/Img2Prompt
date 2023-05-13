#!/bin/bash

#CUDA_VISIBLE_DEVICES='7'
python Imape2Prompt_evaluation.py \
--config ./configs/AOKVQA_caption.yaml --dataset 'aokvqa' \
--output_dir './output_new/vqa_result/' \
--caption_file '../caption_question_files/aokvqa_val_caption.json'  \
--question_file '../caption_question_files/aokvqa_val_question.json' \
--ans_dict_file '../caption_question_files/aokvqa_val_ans_to_cap_dict.json' \
--min_answer_length 0 --max_answer_length 10   \
--dist_selection 'hugging' \
--batch_size_test 1 --evaluate \
--model_selection 'opt-6.7b' --num_caps_per_img 30 --num_question_per_img 30


#--caption_file '/export/home/Big-ZS-VQA/blip_original/output/caption_aokvqa/caption_aokvqa_clip_blk_7_sam_p1_min10_max20_cap120_bt1_npatch20_rp1_t1_topk50_only_10000/result/val_epoch0.json' \
#--question_file '/export/home/Big-ZS-VQA/blip_original/output/qg_generation/tag_aokvqa_gradcamcaption_itm_noadv_3chunk_answer_gradcamcaption_checkpoint_04_10000_val/result/result/output.json'  \
#--ans_dict_file '/export/home/Big-ZS-VQA/blip_original/output/answers/gradcamcaption_itm_noadv_3chunk_aokvqa_itm_rank_10000_val/result/ans_to_cap_dict.json' \
