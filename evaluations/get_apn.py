import torch

def get_apn(inputs, targets, features, idx, t2i, all_idx_l):

    pos = torch.zeros([targets.shape[0], features.shape[1]])
    neg = torch.zeros([targets.shape[0], features.shape[1]])
    anchor = torch.zeros([targets.shape[0], features.shape[1]])
    sim_mat = torch.mm(inputs, features.t().cuda()) 

    for i in range(targets.shape[0]):
        anchor[i] = features[i]
        
        sample = sim_mat[i, :]
        target_iden = str(targets[i].item())
        pos_ind_l = copy.deepcopy(t2i[target_iden])
        neg_ind_l = list(set(all_idx_l) - set(pos_ind_l))
        pos_ind_l.remove(idx[i])
        pos, pos_idx = torch.min(sample[pos_ind_l], dim=0)
        neg, neg_idx = torch.max(sample[neg_ind_l], dim=0)

        pos[i] = features[pos_idx]
        neg[i] = features[neg_idx]

        return anchor, pos, neg