from __future__ import absolute_import, print_function
"""
My WHALES dataset
"""
import torch
import torch.utils.data as data
from PIL import Image

import os
from DataSet import transforms
from collections import defaultdict
import copy


def default_loader(path):
    return Image.open(path).convert('RGB')


def Generate_transform_Dict(origin_width=224, width=224, ratio=0.16):

    std_value = 1.0 / 255.0
    normalize = transforms.Normalize(mean=[104 / 255.0, 117 / 255.0, 128 / 255.0],
                                     std=[1.0/255, 1.0/255, 1.0/255])

    transform_dict = {}

    transform_dict['normalize-only'] = \
        transforms.Compose([
            transforms.CovertBGR(),
            transforms.ToTensor(),
            normalize,
        ])

    transform_dict['rand-crop'] = \
        transforms.Compose([
            transforms.CovertBGR(),
            transforms.Resize(origin_width),
            transforms.RandomResizedCrop(scale=(ratio, 1), size=width),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    transform_dict['center-crop'] = \
        transforms.Compose([
            transforms.CovertBGR(),
            transforms.Resize(origin_width),
            transforms.CenterCrop(width),
            transforms.ToTensor(),
            normalize,
        ])

    transform_dict['resize'] = \
        transforms.Compose([
            transforms.CovertBGR(),
            transforms.Resize(width),
            transforms.CenterCrop(width),
            transforms.ToTensor(),
            normalize,
        ])
    return transform_dict


class MyData(data.Dataset):
    def __init__(self, root=None, label_txt=None,
                 transform=None, loader=default_loader, features=None):

        # Initialization data path and train(gallery or query) txt path
        if root is None:
            self.root = "preprocessed_imgs"
        self.root = root

        if label_txt is None:
            label_txt = os.path.join(root, 'train.txt')

        if transform is None:
            transform_dict = Generate_transform_Dict()['normalize-only']

        # read txt get image path and labels
        file = open(label_txt)
        images_anon = file.readlines()

        images = []
        labels = []

        for img_anon in images_anon:
            # img_anon = img_anon.replace(' ', '\t')

            [img, label] = img_anon.split(' ')
            images.append(img)
            labels.append(int(label))

        classes = list(set(labels))

        # Generate Index Dictionary for every class
        Index = defaultdict(list)
        for i, label in enumerate(labels):
            Index[label].append(i)

        idx_all_l = []
        for i in range(13623):
            idx_all_l.append(i)
     
        t2i = torch.load('drive/My Drive/t2i.pth')['t2i']

        if features == None:
            features = torch.zeros((len(images), len(images)))

        # Initialization Done
        self.root = root
        self.images = images
        self.labels = labels
        self.classes = classes
        self.transform = transform
        self.Index = Index
        self.loader = loader
        self.features = features
        self.idx_all_l = idx_all_l
        self.t2i = t2i

    def __getitem__(self, index):

        anchor = self.features[index]
        anchor.unsqueeze_(0)
        anchor_sim = torch.mm(anchor, self.features.t())
        anchor_sim.squeeze_()

        target_iden = str(self.labels[index])
        pos_ind_l = copy.deepcopy(self.t2i[target_iden])
        neg_ind_l = list(set(self.idx_all_l) - set(pos_ind_l))
        pos_ind_l.remove(index)
        _, pos_idx = torch.min(anchor_sim[pos_ind_l], dim=0)
        pos_ind = pos_ind_l[pos_idx]
        _, neg_idx = torch.max(anchor_sim[neg_ind_l], dim=0)
        neg_ind = neg_ind_l[neg_idx]

        
        pos_fn = self.images[pos_ind]
        neg_fn = self.images[neg_ind]
        fn, label = self.images[index], self.labels[index]
        fn = os.path.join(self.root, fn)
        pos_fn = os.path.join(self.root, pos_fn)
        neg_fn = os.path.join(self.root, neg_fn)

        neg = self.loader(neg_fn)
        pos = self.loader(pos_fn)
        img = self.loader(fn)
        if self.transform is not None:
            pos = self.transform(pos)
            neg = self.transform(neg)
            img = self.transform(img)
        return img, pos, neg, label

    def __len__(self):
        return len(self.images)


class WHALES:
    def __init__(self, width=224, origin_width=256, ratio=0.16, root=None, transform=None, features=None):
        # Data loading code
        # print('ratio is {}'.format(ratio))
        transform_Dict = Generate_transform_Dict(
            origin_width=origin_width, width=width, ratio=ratio)
        if root is None:
            root = "preprocessed_imgs"
            
        train_txt = os.path.join(root, 'train.txt')
        test_txt = os.path.join(root, 'test.txt')

        self.train = MyData(root, label_txt=train_txt,
                            transform=transform_Dict['normalize-only'])
        self.test = MyData(root, label_txt=test_txt,
                              transform=transform_Dict['normalize-only'])


def testWHALES():
    print(WHALES.__name__)
    data = WHALES()
    print(len(data.test))
    print(len(data.train))
    print(data.train[1])


if __name__ == "__main__":
    testWHALES()
