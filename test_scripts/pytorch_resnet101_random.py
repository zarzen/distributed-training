from __future__ import print_function

import torch, torchvision
import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
import horovod.torch as hvd
import tensorboardX
import os
import math
from tqdm import tqdm
from logger import get_logger, log_time, sync_e
import time
from datetime import datetime
import json
import logging
import timeit
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=128,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

allreduce_batch_size = args.batch_size * args.batches_per_allreduce

hvd.init()

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())

cudnn.benchmark = True

# Set up standard ResNet-101 model.
model = models.resnet101()

if args.cuda:
    # Move model to GPU.
    model.cuda()

# Horovod: scale learning rate by the number of GPUs.
# Gradient Accumulation: scale learning rate by batches_per_allreduce
optimizer = optim.SGD(model.parameters(),
                      lr=(args.base_lr *
                          args.batches_per_allreduce * hvd.size()))

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(
    optimizer, named_parameters=model.named_parameters(),
    compression=compression)

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# profile = LineProfiler()

data_batch = torch.randn(args.batch_size, 3, 224, 224)
target_batch = torch.LongTensor(args.batch_size).random_() % 1000
if args.cuda:
    data_batch, target_batch = data_batch.cuda(), target_batch.cuda()

def log(s, nl=True):
    if hvd.rank() != 0:
        return
    print(s, end='\n' if nl else '')

# @profile
def train():
    # model.train()

    optimizer.zero_grad()
    # Split data into sub-batches of size batch_size
    output = model(data_batch)

    loss = F.cross_entropy(output, target_batch)
    # Average gradients among sub-batches
    loss.backward()
    
    # Gradient is applied across all ranks
    optimizer.step()


def log(s, nl=True):
    if hvd.rank() != 0:
        return
    print(s, end='\n' if nl else '')

img_secs = []
for epoch in range(0, args.epochs):
    # train(epoch)
    # validate(epoch)
    # save_checkpoint(epoch)
    timeer_ = timeit.timeit(train, number=20)
    img_sec = args.batch_size * 20 / timeer_
    log('\nIter #%d: %.1f img/sec per GPU in %.1f' % (epoch, img_sec, timeer_))
    img_secs.append(img_sec)

# Results
img_sec_mean = np.mean(img_secs[1:])
img_sec_conf = 1.96 * np.std(img_secs[1:])
log('Img/sec per GPU: %.3f +-%.3f' % (img_sec_mean, img_sec_conf))
log('Total img/sec on %d GPU(s): %.1f +-%.1f' %
    (hvd.size(), hvd.size() * img_sec_mean, hvd.size() * img_sec_conf))

# profile.print_stats()
