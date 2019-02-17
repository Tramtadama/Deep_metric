# coding=utf-8
from __future__ import absolute_import, print_function
import numpy as np
import pandas as pd

def make_whales_predictions(sim_matrix, gallery_lables, new_whale_added=False, new_whale_thrshld=0.5):
    pred_list = []
    whale_inst_pred_list = []
    print(sim_matrix)
    print(sim_matrix.shape)
    for query in tqdm(sim_matrix[0]):
        for i in range(5):
            print(query.shape)
            print(query)
            best_fit_val = np.max(query)
            best_fit_ind = np.argmax(query)
            if (new_whale_added==False) and best_fit_val < new_whale_thrshld:
                new_whale_added = True
                whale_inst_pred_list.append('new_whale')
            else:
                best_fit_id = gallery_lables[best_fit_ind]
                whale_inst_pred_list.append(best_fit_id)
                inds_to_remove = [i for i, x in enumerate(gallery_lables) if x == best_fit_id]
                for ind in inds_to_remove:
                    del query[ind]
        pred_list.append(whale_inst_pred_list)
    return pred_list

def make_whales_sub_file(pred_list):
    test_df = pd.read_csv('sample_submission.csv')
    whale_ids = list(test_df.Image)
    with open('submission.csv', 'wt', newline='\n') as submission_file:
        submission_file.write('Image, Id\n')
        for i, whale_inst_pred_list in enumerate(pred_list):
            whale_id = whale_ids[i]
            s = " "
            pred_string = s.join(whale_inst_pred_list)
            submission_line = whale_id + ',' + pred_string
            submission_file.write(submission_line)