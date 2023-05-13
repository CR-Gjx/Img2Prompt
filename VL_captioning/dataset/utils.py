import re
import torch
import math
import copy
import numpy as np
import random
import os
import json

def prepare_dev_que_ann(dev_set, config, split_seed):
    #inplace modification to config file
    train_ann = json.load(open(config['train_ann_path']))
    train_que = json.load(open(config['train_ques_path']))
    dev_que_id = set([i['question_id'] for i in dev_set])
    train_ann_new = get_dict_subset(train_ann, dev_que_id, key='annotations')
    train_que_new = get_dict_subset(train_que, dev_que_id, key='questions')

    config['fs_dev_ann_path'] = '{}/fs_dev_ann_path_s{}.json'.format(config['output_dir'], split_seed)
    config['fs_dev_que_path'] = '{}/fs_dev_que_path_s{}.json'.format(config['output_dir'], split_seed)
    print('saving few show dev ann and que------')
    json.dump(train_ann_new, open(config['fs_dev_ann_path'], 'w'))
    json.dump(train_que_new, open(config['fs_dev_que_path'], 'w'))

def get_dict_subset(list_of_dict, subset_id, key='annotations'):
    # key = 'annotations' or 'questions'
    new_list_dict = copy.deepcopy(list_of_dict)
    new_list_dict[key] = []
    for i in list_of_dict[key]:
        if i['question_id'] in subset_id:
            new_list_dict[key].append(i)
    return new_list_dict

def split_train_dev(ann_file, num_sample, split_seed=0):
    # num_sample for train and dev independently
    rng = random.Random(split_seed)
    ann = json.load(open(ann_file,'r'))
    ann_agg = aggregate_data(ann)
    k_list = rng.choices(list(ann_agg.keys()), k=2*num_sample)
    total_sample = []
    for k in k_list:
        total_sample.append(rng.choice(ann_agg[k]))

    train_set = copy.deepcopy(total_sample[:num_sample])
    dev_set = copy.deepcopy(total_sample[num_sample:])

    return train_set, dev_set

def aggregate_data(data):
    data_new_dict = {}
    for i in data:
        image_id = i['image'].split('_')[-1].split('.')[0].lstrip('0')
        if image_id in data_new_dict.keys():
            data_new_dict[image_id].append(i)
        else:
            data_new_dict[image_id] = []
            data_new_dict[image_id].append(i)

    return data_new_dict

def encoder_det_label(boxes, labels, num_cls=80, bbox_area=1, centre_grid=False):
    '''
    boxes (np.array) [[x1,y1,x2,y2],[]], normalised, np.array
    labels (np.array) [...]
    return
    target (num_patch, num_patch, total vocab size) (tensor)
    return 7x7x91 (actually just 80 classes, just use category as cls as example)
    make sure labels is from zero index
    '''
    grid_num = 7
    if len(boxes) == 0:
        return torch.zeros((grid_num, grid_num, num_cls))
    else:
        assert (boxes <= 1).all()


        num_boxes = boxes.shape[0]
        target = torch.zeros((grid_num, grid_num, num_cls))
        cell_size = 1. / grid_num
        wh = boxes[:, 2:] - boxes[:, :2]

        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2

        wh_offset = (wh * np.sqrt(bbox_area)) / 2
        new_x1y1 = cxcy - wh_offset
        new_x2y2 = cxcy + wh_offset

        for i in range(num_boxes):
            cxcy_sample = cxcy[i]

            ij = np.floor(cxcy_sample / cell_size)

            if centre_grid:
                target[int(ij[1]), int(ij[0]), int(labels[i])] = 1
            else:
                new_x1y1_sample = new_x1y1[i]
                new_x2y2_sample = new_x2y2[i]

                ind_x1y1 = np.floor(new_x1y1_sample / cell_size)
                ind_x2y2 = np.floor(new_x2y2_sample / cell_size)
                diff_ind = ind_x2y2 - ind_x1y1 > 0

                # add +1 to ensure that that cell is included in target label
                if diff_ind.all():
                    target[int(ind_x1y1[1]):int(ind_x2y2[1] + 1), int(ind_x1y1[0]):int(ind_x2y2[0] + 1), int(labels[i])] = 1
                elif diff_ind[0]:
                    target[int(ij[1]), int(ind_x1y1[0]):int(ind_x2y2[0] + 1), int(labels[i])] = 1
                elif diff_ind[1]:
                    target[int(ind_x1y1[1]):int(ind_x2y2[1] + 1), int(ij[0]), int(labels[i])] = 1
                else:
                    target[int(ij[1]), int(ij[0]), int(labels[i])] = 1

        return target

def corrupt_spans(text, rng, mask_ratio=0.15):
    """T5-style Masked Language Modeling with corrupted span prediction
    Args:
        text

    Returns:
        source_text (masked_text)
        target_text

    Ex) (in vocab ids)
    input
        In this tutorial, we’ll explore how to preprocess your data using Transformers. The main tool for this is what we call a tokenizer.

    masked_text
        <extra_id_0> this tutorial, we’ll explore how to preprocess your data <extra_id_1> Transformers. The main tool for this is what <extra_id_2> call a tokenizer.
    target_text
    """

    tokens = text.split()

    n_tokens = len(tokens)

    n_mask = int(max(mask_ratio * n_tokens, 1))

    mask_indices = list(range(n_tokens))
    rng.shuffle(mask_indices)
    mask_indices = sorted(mask_indices[:n_mask])

    assert len(mask_indices) > 0, text

    span = [mask_indices[0], mask_indices[0] + 1]
    spans = []

    for i, mask_index in enumerate(mask_indices):
        # if current mask is not the last one & the next mask is right after current mask
        if i < len(mask_indices) - 1 and mask_indices[i + 1] == mask_index + 1:
            contiguous = True
        else:
            contiguous = False

        if contiguous:
            span[1] += 1

        else:
            # non contiguous -> output current span
            spans.append(span)
            # if current mask is not the last one -> create next span
            if i < len(mask_indices) - 1:
                span = [mask_indices[i + 1], mask_indices[i + 1] + 1]

    masked_tokens = copy.deepcopy(tokens)

    target_tokens = []
    cum_span_length = 0
    for i, span in enumerate(spans):
        start, end = span

        masked_tokens[start - cum_span_length + i: end -
                                                   cum_span_length + i] = [f'<extra_id_{i}>']

        target_tokens.append(f'<extra_id_{i}>')
        target_tokens.extend(tokens[start:end])

        cum_span_length += (end - start)
    target_tokens.append(f'<extra_id_{i + 1}>')
    masked_text = " ".join(masked_tokens)

    target_text = " ".join(target_tokens)

    return masked_text, target_text

def pre_caption_label_sep(caption, max_words, label=None):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        ' ',
        caption.lower(),
    ).replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')

    if len(caption_words) > max_words:
        caption = caption_words[:max_words]
        label = label[:max_words]
    else:
        caption = caption_words
    caption_words_len = len(caption)

    return caption, caption_words_len

def pre_noun_caption_sep(caption, max_words, remove_dup=True):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        ' ',
        caption.lower(),
    ).replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    # the list will become unorder after set, it might be okay if we only keep the noun,
    # but if we only remove stop word, maybe don't use set to retain semantics
    if remove_dup:
        caption_words = list(set(caption.split(' ')))
    else:
        caption_words = caption.split(' ')

    if len(caption_words) > max_words:
        caption = caption_words[:max_words]
    else:
        caption = caption_words
    caption_words_len = len(caption)

    return caption, caption_words_len

def pre_noun_caption(caption, max_words, remove_dup=True):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        ' ',
        caption.lower(),
    ).replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    if remove_dup:
        caption_words = list(set(caption.split(' ')))
    else:
        caption_words = caption.split(' ')

    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])
    else:
        caption = ' '.join(caption_words)

    return caption

def prefix_lm(caption, max_words, rng=None, min_val=0.1, max_val=0.5):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        ' ',
        caption.lower(),
    ).replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    caption_words_len = len(caption_words)

    if caption_words_len > max_words:
        caption_words = caption_words[:max_words]
        caption = ' '.join(caption_words)
        caption_words_len = max_words

    # split into source and target caption
    try:
        src_len = rng.randrange(math.ceil(min_val*caption_words_len), math.ceil(max_val*caption_words_len))
    except:
        # this happens if caption len is too short, ensure at least there are min 1 word in src and tgt cap
        src_len = rng.randrange(1, caption_words_len)
    src_caption = ' '.join(caption_words[:src_len])
    tgt_caption = ' '.join(caption_words[src_len:])

    return caption, src_caption, tgt_caption

def pre_question(question,max_ques_words):
    question = re.sub(
        r"([.!\"()*#:;~])",
        '',
        question.lower(),
    ).replace('-', ' ').replace('/', ' ')  
    question = question.rstrip(' ')
    
    #truncate question
    question_words = question.split(' ')
    if len(question_words)>max_ques_words:
        question = ' '.join(question_words[:max_ques_words])
            
    return question

def pre_caption_min(caption):
    caption = re.sub(
        r"([.])",
        ' ',
        caption.lower(),
    ).replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    return caption

def pre_caption(caption, max_words):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        ' ',
        caption.lower(),
    ).replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption

from vqaTools.vqaEval import VQAEval
from vqaTools.gqaEval import GQAEval
import json
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

import utils
from tqdm import tqdm


def vqa_eval(vqa, result_file, test_ques_path, logger=None, n=2, dataset='vqa'):

    vqaRes = vqa.loadRes(result_file, test_ques_path)
    # create vqaEval object by taking vqa and vqaRes
    vqaEval = VQAEval(vqa, vqaRes, n=n)  # n is precision of accuracy (number of places after decimal), default is 2
    # evaluate results
    if dataset != 'aokvqa':
        vqaEval.evaluate()
    else:
        vqaEval.evaluate_aokvqa()




    if logger is None:
        # print accuracies
        print("\n")
        print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
        print("Per Answer Type Accuracy is the following:")
        for ansType in vqaEval.accuracy['perAnswerType']:
            print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
        print("\n")
    else:
        # print accuracies
        logger.info("\n")
        logger.info("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
        logger.info("Per Answer Type Accuracy is the following:")
        for ansType in vqaEval.accuracy['perAnswerType']:
            logger.info("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
        logger.info("\n")
    return vqaEval


def broadcast_result(result, result_dir, filename, is_json=True, is_list=True):
    if is_json:
        result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
        final_result_file = os.path.join(result_dir, '%s.json'%filename)
        json.dump(result,open(result_file,'w'))
    else:
        result_file = os.path.join(result_dir, '%s_rank%d.pth'%(filename,utils.get_rank()))
        final_result_file = os.path.join(result_dir, '%s.pth'%filename)
        torch.save(result,result_file)     
        
    dist.barrier()

    if is_list:
        result = []
    else:
        result = {}
    for rank in range(utils.get_world_size()):
        if is_json:
            result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
            res = json.load(open(result_file,'r'))
        else:
            result_file = os.path.join(result_dir, '%s_rank%d.pth'%(filename,rank))
            res = torch.load(result_file)            
        if is_list:
            result += res
        else:
            result.update(res) 

    return result 

    
def collect_result(result, result_dir, filename, is_json=True, is_list=True):
    if is_json:
        result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
        final_result_file = os.path.join(result_dir, '%s.json'%filename)
        json.dump(result,open(result_file,'w'))
    else:
        result_file = os.path.join(result_dir, '%s_rank%d.pth'%(filename,utils.get_rank()))
        final_result_file = os.path.join(result_dir, '%s.pth'%filename)
        torch.save(result,result_file)     
        
    dist.barrier()
    
    result = None
    if utils.is_main_process():   
        # combine results from all processes
        if is_list:
            result = []
        else:
            result = {}
        for rank in range(utils.get_world_size()):
            if is_json:
                result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
                res = json.load(open(result_file,'r'))
            else:
                result_file = os.path.join(result_dir, '%s_rank%d.pth'%(filename,rank))
                res = torch.load(result_file)            
            if is_list:
                result += res
            else:
                result.update(res) 
      
    return result    

    
def save_result(result, result_dir, filename, is_json=True, is_list=True, remove_duplicate='', distributed=False):

    if is_json:
        result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
        final_result_file = os.path.join(result_dir, '%s.json'%filename)
        json.dump(result,open(result_file,'w'))
    else:
        result_file = os.path.join(result_dir, '%s_rank%d.pth'%(filename,utils.get_rank()))
        final_result_file = os.path.join(result_dir, '%s.pth'%filename)
        torch.save(result,result_file)
    if distributed:
        dist.barrier()


    if utils.is_main_process():   
        # combine results from all processes
        if is_list:
            result = []
        else:
            result = {}
        for rank in range(utils.get_world_size()):
            if is_json:
                result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
                res = json.load(open(result_file,'r'))
            else:
                result_file = os.path.join(result_dir, '%s_rank%d.pth'%(filename,rank))
                res = torch.load(result_file)            
            if is_list:
                result += res
            else:
                result.update(res)
                
        id_list = []       
        
        if remove_duplicate:
            result_new = []
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new             
                
        if is_json:                  
            json.dump(result,open(final_result_file,'w'))   
        else:            
            torch.save(result,final_result_file)     
        
        print('result file saved to %s'%final_result_file)
    # if distributed:
    #     dist.barrier()
    return final_result_file


# IoU function
def computeIoU(box1, box2):
    # each box is of [x1, y1, w, h]
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[0]+box1[2]-1, box2[0]+box2[2]-1)
    inter_y2 = min(box1[1]+box1[3]-1, box2[1]+box2[3]-1)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = box1[2]*box1[3] + box2[2]*box2[3] - inter
    return float(inter)/union
        
        
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

def caption_eval(annotation_file, results_file):

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
    
    return coco_eval
