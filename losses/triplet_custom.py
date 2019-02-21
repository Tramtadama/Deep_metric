from torch import nn

class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets, sim_mat, idx, t2i, all_idx_l):

        for i in range(targets.shape[0]):
            sample = sim_mat[idx[i], :]
            target_iden = str(targets[i])
            pos_ind_l = t2i[target_iden]
            pos_ind_l.remove(idx)
            pos = torch.min(sample[pos_ind_l])
            neg_ind_l = list(set(all_idx_l) - set(pos_ind_l))
            neg = torch.max(sample[neg_ind_l])
            loss_formula = pos - neg + margin
            loss = max(loss_formula, 0)
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
        return loss.sum(dim=1).mean()