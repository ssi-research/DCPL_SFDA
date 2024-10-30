"""
 This file was copied from https://github.com/tim-learn/SHOT-plus/code/uda/data_list.py and modified for this project needs.
 The license of the file is in: https://github.com/tim-learn/SHOT-plus/blob/master/LICENSE
"""


import os
import os.path

import torch
import torch.utils.data
from torch.utils.data import Dataset

from PIL import Image
import os.path


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def make_dataset(image_list, labels):
    if labels == 0:
        images = image_list
    else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, label_transform=None, img_root_dir=None, useDict=True):
        self.img_root_dir = img_root_dir
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders \n"))

        self.imgs = imgs
        self.transform = transform
        self.label_transform = label_transform
        self.loader = rgb_loader

    def __getitem__(self, index):

        img_name, label = self.imgs[index]
        if self.img_root_dir is not None:
            img_name = os.path.join(self.img_root_dir, img_name)

        img = self.loader(img_name)
        img_tr = self.transform(img)

        return img_tr, label, img_name, index

    def __len__(self):
        return len(self.imgs)


################################################
def org_make_dataset(image_list, labels):
    if labels is not None:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


class OrgImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = org_make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders \n"))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)



