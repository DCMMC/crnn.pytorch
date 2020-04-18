from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utils
# import dataset
from dataset import alignCollate
from dataset import THUCNewsDataset
from dataset import randomSequentialSampler
import json
# DCMMC: for get stroke count for chinese char
# TODO: only support py2
# from cjklib.characterlookup import CharacterLookup
from Levenshtein import distance as levenshtein_distance

import models.crnn as crnn

# DCMMC: for get stroke count for chinese char
# cjk = CharacterLookup('C')
parser = argparse.ArgumentParser()
parser.add_argument('--trainRoot', required=True, help='path to dataset')
parser.add_argument('--valRoot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
# TODO(meijieru): epoch -> iter
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
parser.add_argument('--expr_dir', default='expr', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, not used by adadealta')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.expr_dir):
    os.makedirs(opt.expr_dir)

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# train_dataset = dataset.lmdbDataset(root=opt.trainroot)
train_dataset = THUCNewsDataset(root=opt.trainRoot, debug=True)
assert train_dataset
if not opt.random_sample:
    sampler = randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=False, sampler=sampler,
    num_workers=int(opt.workers),
    collate_fn=alignCollate(
        imgH=opt.imgH, imgW=opt.imgW,
        keep_ratio=opt.keep_ratio))
# test_dataset = dataset.lmdbDataset(
#     root=opt.valroot, transform=dataset.resizeNormalize((100, 32)))
test_dataset = THUCNewsDataset(root=opt.valRoot, mode='val')

# DCMMC: alphabet from file
if opt.alphabet.endswith('.json'):
    with open(opt.alphabet, 'r') as f:
        opt.alphabet = json.load(f)
    if isinstance(opt.alphabet, dict):
        opt.alphabet = list(opt.alphabet.keys())
    assert isinstance(opt.alphabet, list)
    # sort by stroke count
    # opt.alphabet = [[c, cjk.getStrokeCount(c)] for c in opt.alphabet]
    # opt.alphabet = sorted(opt.alphabet, key=lambda c: c[1])
    # opt.alphabet = ''.join([c[0] for c in opt.alphabet])
    opt.alphabet = ''.join(opt.alphabet)
    print('First chars of alphabet:', opt.alphabet[:30])
nclass = len(opt.alphabet) + 1
# grayimage
nc = 1

converter = utils.strLabelConverter(opt.alphabet)
criterion = CTCLoss()


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


crnn = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
crnn.apply(weights_init)
if opt.pretrained != '':
    print('loading pretrained model from %s' % opt.pretrained)
    # DCMMC: encounter with DataParallel
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
    # original saved file with DataParallel
    state_dict = torch.load(opt.pretrained)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('modules.') else k # remove `module.`
        new_state_dict[name] = v
        # load params
    crnn.load_state_dict(new_state_dict)
    # crnn.load_state_dict(torch.load(opt.pretrained))
print(crnn)

device = torch.device("cuda:0")
# image = torch.FloatTensor(opt.batchSize, 1, opt.imgH, opt.imgH)
# text = torch.IntTensor(opt.batchSize * 5)
# length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    crnn.cuda()
    if opt.ngpu > 1:
        crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    # image = image.cuda()
    criterion = criterion.cuda()

# image = Variable(image)
# text = Variable(text)
# length = Variable(length)

# loss averager
loss_avg = utils.averager()
# distance averager
distance_avg = 0.

# setup optimizer
if opt.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn.parameters())
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)


def levenshtein_distance_norm(str1, str2):
    '''
    归一化的编辑距离, 越小越好
    @return [0, 1]
    '''
    max_len = max(len(str1), len(str2), 1)
    return levenshtein_distance(str1, str2) / max_len


def evalBatch(net, criterion):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    print('cpu_images', cpu_images.size())
    print('cpu_texts', len(cpu_texts), max([len(i) for i in cpu_texts]))
    print(sorted(list(enumerate(cpu_texts)), key=lambda x: len(x[1]))[-1])
    batch_size = cpu_images.size(0)
    # utils.loadData(image, cpu_images)
    text, length = converter.encode(cpu_texts)
    # t, l = converter.encode(cpu_texts)
    # utils.loadData(text, t)
    # utils.loadData(length, l)
    image = cpu_images.to(device)
    # text = text.to(device)
    # length = length.to(device)

    # print('image:', image.size())
    preds = crnn(image)
    # print('shape, size of preds:', preds.shape)
    # preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    preds_size = torch.IntTensor([preds.size(0)] * batch_size)
    loss = criterion(preds, text, preds_size, length) / batch_size
    _, preds = preds.max(2)
    # print('shape, size of preds:', preds.shape)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    # print('shape, size of flatten preds:', preds.shape)
    sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
    distance = 0.
    cnt = 0
    for pred, target in zip(sim_preds, cpu_texts):
        distance += levenshtein_distance_norm(pred, target)
        if cnt < 1:
            print('pred:\n{}\ntrue:\n{}\n'.format(pred, target))
            print('*'*50)
            cnt += 1
    distance /= len(sim_preds)
    return loss, distance


train_iter = iter(train_loader)
i = 0
while i < len(train_loader):
    for p in crnn.parameters():
        p.requires_grad = False
    crnn.eval()

    cost, distance = evalBatch(crnn, criterion)
    loss_avg.add(cost)
    distance_avg += distance
    i += 1

    if i % opt.displayInterval == 0:
        print('[%d/%d] Loss: %f, distance: %f' %
              (i, len(train_loader), loss_avg.val(), distance_avg / i))
        # loss_avg.reset()
