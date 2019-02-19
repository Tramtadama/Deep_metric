# coding=utf-8
from __future__ import absolute_import, print_function
import pandas as pd
import torch
import copy

def make_whales_predictions(sim_matrix, gallery_lables, new_whale_added=False, new_whale_thrshld=0.8):
    label_ids = torch.load('drive/My Drive/labels_ids.pth')['label_ids']
    pred_list = []
    whale_inst_pred_list = []
    for query_ind in range(sim_matrix.shape[0]):
        query = sim_matrix[query_ind]
        gallery_copy = copy.deepcopy(gallery_lables)
        for i in range(5):
            best_fit_val, best_fit_ind = torch.max(query, dim=0)
            if (new_whale_added==False) and best_fit_val < new_whale_thrshld:
                new_whale_added = True
                whale_inst_pred_list.append('new_whale')
            else:
                best_fit_id = gallery_copy[best_fit_ind]
                whale_id_string = label_ids[best_fit_id][0]
                whale_inst_pred_list.append(whale_id_string)
                inds_to_remove = [k for k, x in enumerate(gallery_copy) if x == best_fit_id]
                for j, ind in enumerate(inds_to_remove):
                    ind -= j
                    del gallery_copy[ind]
                    query = torch.cat([query[:ind], query[ind+1:]])
        pred_list.append(whale_inst_pred_list)
        whale_inst_pred_list = []
    return pred_list

def make_whales_sub_file(pred_list):
    test_df = pd.read_csv('sample_submission.csv')
    whale_ids = list(test_df.Image)
    with open('submission.csv', 'wt', newline='\n') as submission_file:
        submission_file.write('Image,Id\n')
        for i, whale_inst_pred_list in enumerate(pred_list):
            whale_id = whale_ids[i]
            s = " "
            pred_string = s.join(whale_inst_pred_list)
            submission_line = whale_id + ',' + pred_string
            submission_file.write(submission_line + '\n')