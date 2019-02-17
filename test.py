# coding=utf-8
from __future__ import absolute_import, print_function
import argparse
from Model2Feature import Model2Feature
import DataSet
import models
from evaluations import Recall_at_ks, pairwise_similarity, extract_features
from utils.serialization import load_checkpoint
from whales_sub import make_whales_sub_file, make_whales_predictions
import torch
import ast

parser = argparse.ArgumentParser(description='PyTorch Testing')

parser.add_argument('--data', type=str, default='cub')
parser.add_argument('--whales', type=ast.literal_eval, default=False)
parser.add_argument('--data_root', type=str, default=None)
parser.add_argument('--gallery_eq_query', '-g_eq_q', type=ast.literal_eval, default=False,
                    help='Is gallery identical with query')
parser.add_argument('--net', type=str, default='VGG16-BN')
parser.add_argument('--resume', '-r', type=str,
                    default='model.pkl', metavar='PATH')

parser.add_argument('--dim', '-d', type=int, default=512,
                    help='Dimension of Embedding Feather')
parser.add_argument('--width', type=int, default=224,
                    help='width of input image')

parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--nThreads', '-j', default=16, type=int, metavar='N',
                    help='number of data loading threads (default: 2)')
parser.add_argument('--pool_feature', type=ast.literal_eval, default=False, required=False,
                    help='if True extract feature from the last pool layer')

args = parser.parse_args()

checkpoint = load_checkpoint(args.resume)
print(args.pool_feature)
epoch = checkpoint['epoch']


if args.gallery_eq_query is True:
    gallery_feature, gallery_labels, query_feature, query_labels = \
        Model2Feature(data=args.data, root=args.data_root, width=args.width, net=args.net, checkpoint=checkpoint,
                      dim=args.dim, batch_size=args.batch_size, nThreads=args.nThreads, pool_feature=args.pool_feature)
    sim_mat = pairwise_similarity(query_feature, gallery_feature)
    sim_mat = sim_mat - torch.eye(sim_mat.size(0))
else:
    model = models.create(args.net, dim=args.dim, pretrained=False)
    resume = load_checkpoint(args.resume)
    model.load_state_dict(resume['state_dict'])
    model = torch.nn.DataParallel(model).cuda()

    data = DataSet.create(args.data, width=args.width, root=args.data_root)

    gallery_loader = torch.utils.data.DataLoader(
        data.gallery, batch_size=args.batch_size, shuffle=False,
        drop_last=False, pin_memory=True, num_workers=args.nThreads)

    query_loader = torch.utils.data.DataLoader(
        data.train, batch_size=args.batch_size,
        shuffle=False, drop_last=False,
        pin_memory=True, num_workers=args.nThreads)

    gallery_feature, gallery_labels = extract_features(
        model, gallery_loader, print_freq=1e5, metric=None, pool_feature=args.pool_feature)
    query_feature, query_labels = extract_features(
        model, query_loader, print_freq=1e5, metric=None, pool_feature=args.pool_feature)

    sim_mat = pairwise_similarity(query_feature, gallery_feature)

if args.whales==True:
    whales_preds = make_whales_predictions(sim_mat, gallery_labels)
    make_whales_sub_file(whales_preds)
else:
    recall_ks = Recall_at_ks(sim_mat, query_ids=query_labels,
                            gallery_ids=gallery_labels, data=args.data)

    result = '  '.join(['%.4f' % k for k in recall_ks])

print('Epoch-%d' % epoch, result)
