#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image, ImageOps
import numpy as np
import h5py


class THUCNewsDataset(Dataset):
    '''
    A synthetic data generated from THUCNews
    @root path of the h5 file
    @mode train/val
    @train_ratio the ratio of training samples in the dataset
    '''

    def __init__(self, root, mode='train', train_ratio=0.8, debug=False):
        self.dataset = h5py.File(root, 'r')
        self.length = len(self.dataset)
        self.offset = 0 if mode == 'train' else int(self.length * train_ratio)
        self.size = int(self.length * train_ratio) if mode == 'train' else \
            (self.length - self.offset)
        self.mode = mode
        if mode == 'val':
            self.toTensor = transforms.ToTensor()
        if debug:
            self.length = 6400
            self.offset = 0 if mode == 'train' else 5600
            self.size = 5600 if mode == 'train' else 800

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        index = int(index.item()) if isinstance(index, torch.Tensor) else \
            int(index)
        assert index <= len(self), 'index range error'
        img = self.dataset[str(index + self.offset)]['img'][...]
        img = Image.fromarray(img)
        label = str(self.dataset[str(index + self.offset)]['y'][...])
        # remove \xa0, \u30000, whitespace...
        label = ''.join(label.strip().split())
        return (img, label)


class lmdbDataset(Dataset):

    def __init__(self, root=None, transform=None, target_transform=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'))
            self.nSamples = nSamples

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            if self.transform is not None:
                img = self.transform(img)

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key))

            if self.target_transform is not None:
                label = self.target_transform(label)

        return (img, label)


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.ANTIALIAS):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        # img = img.resize(self.size, self.interpolation)
        # DCMMC: keep aspect ratio and padding with white
        ratio = self.size[1] / img.size[1]
        img.thumbnail([s * ratio for s in img.size], self.interpolation)
        delta_w = self.size[0] - img.size[0]
        img = ImageOps.expand(
            img, (delta_w // 2, 0, delta_w - delta_w // 2, 0),
            fill=255)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            # DCMMC
            # batch_index = random_start + torch.range(0, self.batch_size - 1)
            batch_index = random_start + torch.arange(0, self.batch_size)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            # tail_index = random_start + torch.range(0, tail - 1)
            tail_index = random_start + torch.arange(0, tail)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=True, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        ratios = []
        for image in images:
            w, h = image.size
            ratios.append(w / float(h))
        # ratios.sort()
        # max_ratio = ratios[-1]
        # imgW = int(np.floor(max_ratio * imgH))
        # imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW
        max_ratio = max(ratios)
        imgW_new = int(np.ceil(max_ratio * imgH))
        transform = resizeNormalize((imgW_new, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels
