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

#--question_file '/export/home/Big-ZS-VQA/blip_original/output/qg_generation/tag_okvqa_gradcamcaption_itm_noadv_3chunk_answer_40gradcamcaption_checkpoint_04_10000/result/result/output.json'  \
#--ans_dict_file '/export/home/Big-ZS-VQA/blip_original/output/answers/gradcamcaption_itm_noadv_3chunk_okvqa_itm_rank_10000/result/ans_to_cap_dict.json' \

#--caption_file '/export/home/Big-ZS-VQA/blip_original/output/caption_okvqa_que_gradcam_itm_rank_new/Caption_okvqa_que_gradcam_itm_rank_blk_7_sam_p1_min10_max20_cap100_bt1_npatch20_rp1_t1_topk50_only_10000/result/val_epoch0.json' \
#--question_file '../generate_questions/result/okvqa/result/output.json'  \
#--ans_dict_file '../generate_questions/result/okvqa/result/ans_to_cap_dict.json' \
