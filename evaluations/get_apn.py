import torch
import copy
import pdb


def get_apn(inputs, targets, features, idx, t2i, all_idx_l):

    pdb.set_trace()
    pos = torch.zeros([targets.shape[0], features.shape[1]])
    neg = torch.zeros([targets.shape[0], features.shape[1]])
    anchor = torch.zeros([targets.shape[0], features.shape[1]])
    sim_mat = torch.mm(inputs, features.t().cuda())

    for i in range(targets.shape[0]):
        ind = idx[i]
        anchor[i] = features[ind]
        sample = sim_mat[i, :]
        target_iden = str(targets[i].item())
        pos_ind_l = copy.deepcopy(t2i[target_iden])
        neg_ind_l = list(set(all_idx_l) - set(pos_ind_l))
        pos_ind_l.remove(ind)
        _, pos_idx = torch.min(sample[pos_ind_l], dim=0)
        pos_ind = pos_ind_l[pos_idx]
        _, neg_idx = torch.max(sample[neg_ind_l], dim=0)
        neg_ind = neg_ind_l[neg_idx]
        pos[i] = features[pos_ind]
        neg[i] = features[neg_ind]

    pdb.set_trace()   
    return anchor, pos, neg
