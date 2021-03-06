from torch import nn
import torch
import torch.nn.functional as F
import copy
import pdb

class TripletLoss(nn.Module):
    def __init__(self, margin=0, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets, features, idx, t2i, all_idx_l):

        sim_mat = torch.mm(inputs, features.t().cuda())
        n = targets.shape[0]
        loss = []

        for i in range(targets.shape[0]):
            sample = sim_mat[i, :]
            target_iden = str(targets[i].item())
            pos_ind_l = copy.deepcopy(t2i[target_iden])
            neg_ind_l = list(set(all_idx_l) - set(pos_ind_l))

            if idx[i] not in pos_ind_l:
                pdb.set_trace()
            pos_ind_l.remove(idx[i])
            if pos_ind_l == []:
                pdb.set_trace()
            pos, _ = torch.min(sample[pos_ind_l], dim=0)
            neg, _ = torch.max(sample[neg_ind_l], dim=0)
            check = neg + self.margin
            loss_formula = F.relu(neg - pos + self.margin)
            loss.append(loss_formula)
        #take sim_mat[idx[0], :]
        #get sim_mat[idx[0], :] cols_target = check what cols correspond to same class as target except for the anchor(need target to idx map) and
        #positive_instances = cols_target
        #negative_instances = rest of sim_mat[idx[0], :]
        #choose negative
        #get max of negative_instances
        #choose positive
        #get min of positive_instances
        #formula for loss
        #loss = max(pos-neg+margin,0)
        pdb.set_trace()
        batch_loss = sum(loss)/n
        return batch_loss