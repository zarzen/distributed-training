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
torch.manual_seed(args.seed)

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)

cudnn.benchmark = True

# If set > 0, will resume training from a given checkpoint.
resume_from_epoch = 0
for try_epoch in range(args.epochs, 0, -1):
    if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
        resume_from_epoch = try_epoch
        break

# Horovod: broadcast resume_from_epoch from rank 0 (which will have
# checkpoints) to other ranks.
resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                                  name='resume_from_epoch').item()

# Horovod: write TensorBoard logs on first worker.
# log_writer = tensorboardX.SummaryWriter(args.log_dir) if hvd.rank() == 0 else None


# Horovod: limit # of CPU threads to be used per worker.
__n_threads = int(os.cpu_count() / torch.cuda.device_count())
print('torch num threads:', __n_threads)
torch.set_num_threads(__n_threads)

kwargs = {'num_workers': __n_threads, 'pin_memory': True} if args.cuda else {}
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

print('cwd:', os.getcwd())
train_dataset = torchvision.datasets.CIFAR10(
    root='~/distributed-training/data', train=True,
    download=False, transform=transform)

# Horovod: use DistributedSampler to partition data among workers. Manually specify
# `num_replicas=hvd.size()` and `rank=hvd.rank()`.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=allreduce_batch_size,
    sampler=train_sampler, **kwargs)



# Set up standard ResNet-101 model.
model = models.resnet101()

if args.cuda:
    # Move model to GPU.
    model.cuda()

# Horovod: scale learning rate by the number of GPUs.
# Gradient Accumulation: scale learning rate by batches_per_allreduce
optimizer = optim.SGD(model.parameters(),
                      lr=(args.base_lr *
                          args.batches_per_allreduce * hvd.size()),
                      momentum=args.momentum, weight_decay=args.wd)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(
    optimizer, named_parameters=model.named_parameters(),
    compression=compression)

# Restore from a previous checkpoint, if initial_epoch is specified.
# Horovod: restore on the first worker which will broadcast weights to other workers.
if resume_from_epoch > 0 and hvd.rank() == 0:
    filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

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
for epoch in range(resume_from_epoch, args.epochs):
    # train(epoch)
    # validate(epoch)
    # save_checkpoint(epoch)
    timeer_ = timeit.timeit(train, number=len(train_loader))
    img_sec = args.batch_size * len(train_loader) / timeer_
    log('\nIter #%d: %.1f img/sec per GPU in %.1f' % (epoch, img_sec, timeer_))
    img_secs.append(img_sec)

# Results
img_sec_mean = np.mean(img_secs[1:])
img_sec_conf = 1.96 * np.std(img_secs[1:])
log('Img/sec per GPU: %.3f +-%.3f' % (img_sec_mean, img_sec_conf))
log('Total img/sec on %d GPU(s): %.1f +-%.1f' %
    (hvd.size(), hvd.size() * img_sec_mean, hvd.size() * img_sec_conf))

# profile.print_stats()
