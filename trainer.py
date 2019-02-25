# coding=utf-8
from __future__ import print_function, absolute_import
import time
from utils import AverageMeter, orth_reg
import  torch
from torch.autograd import Variable
from torch.backends import cudnn

cudnn.benchmark = True


def train(epoch, model, criterion, optimizer, train_loader, args, features, idx_all_l):

    losses = AverageMeter()
    batch_time = AverageMeter()
    accuracy = AverageMeter()
    pos_sims = AverageMeter()
    neg_sims = AverageMeter()

    end = time.time()

    freq = min(args.print_freq, len(train_loader))

    for i, data_ in enumerate(train_loader, 0):

        inputs, poss, negs, labels = data_

        inputs = inputs.cuda()
        labels = labels.cuda()
        poss = poss.cuda()
        negs = negs.cuda()

        optimizer.zero_grad()

        anchor_emb = model(inputs)
        poss_emb = model(poss)
        negs_emb = model(negs)

        loss = criterion(anchor_emb, poss_emb, negs_emb)

        if args.orth_reg != 0:
            loss = orth_reg(net=model, loss=loss, cof=args.orth_reg)

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.data.item())


        if (i + 1) % freq == 0 or (i+1) == len(train_loader):
            print('Epoch: [{0:03d}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f} \t'.format
                  (epoch + 1, i + 1, len(train_loader), batch_time=batch_time,
                   loss=losses,))

        if epoch == 0 and i == 0:
            print('-- HA-HA-HA-HA-AH-AH-AH-AH --')
