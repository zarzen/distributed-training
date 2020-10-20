from __future__ import print_function

import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import timeit
import numpy as np

import sys
import torch

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.multiprocessing import Pool, Process

# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')

parser.add_argument('--url-port', type=str, default='tcp://172.31.22.234:23456',
                    help='tcp://172.31.22.234:23456')

parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')

parser.add_argument('--world-size', type=int, default=32,
                    help='world size')

parser.add_argument('--num-gpu', type=int, default=1,
                    help='num of gpu')

parser.add_argument('--world-id', type=int, default=0,
                    help='world id')

parser.add_argument('--local_rank', type=int, default=0,
                    help='local id')

parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')

args = parser.parse_args()

cudnn.benchmark = True

# Number of additional worker processes for dataloading 数据加载的额外工作进程数
workers = 2

# Number of distributed processes 分布式进程数
world_size = args.world_size

# Distributed backend type 分布式后端类型
dist_backend = 'nccl'

# Url used to setup distributed training 设置分布式训练的 url
dist_url = args.url_port

# Initialize Process Group 初始化进程组
# v1 - init with url  使用 url 初始化
dist.init_process_group(backend=dist_backend, init_method=dist_url, rank=int(args.world_id * args.num_gpu + args.local_rank), world_size=world_size)
# v2 - init with file 使用文件初始化
# dist.init_process_group(backend="nccl", init_method="file:///home/ubuntu/pt-distributed-tutorial/trainfile", rank=int(sys.argv[1]), world_size=world_size)
# v3 - init with environment variables 使用环境变量初始化
# dist.init_process_group(backend="nccl", init_method="env://", rank=int(sys.argv[1]), world_size=world_size)

# Establish Local Rank and set device on this node 设置节点的本地化编号和设备
local_rank = int(args.local_rank)
dp_device_ids = [local_rank]
torch.cuda.set_device(local_rank)

# Set up standard model.
model = getattr(models, args.model)().cuda()

model = torch.nn.parallel.DistributedDataParallel(model, device_ids=dp_device_ids, output_device=local_rank)

optimizer = optim.SGD(model.parameters(), lr=0.01)

# Set up fixed fake data
data = torch.randn(args.batch_size, 3, 224, 224)
target = torch.LongTensor(args.batch_size).random_() % 1000
data, target = data.cuda(), target.cuda()


def benchmark_step():
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()


def log(s, nl=True):
    if args.world_id != 0 or args.local_rank != 0:
        return
    print(s, end='\n' if nl else '')


log('Model: %s' % args.model)
log('Batch size: %d' % args.batch_size)
device = 'GPU'
log('Number of %ss: %d' % (device, args.world_size))

# Warm-up
log('Running warmup...')
timeit.timeit(benchmark_step, number=args.num_warmup_batches)

# Benchmark
log('Running benchmark...')
img_secs = []
for x in range(args.num_iters):
    time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
    img_sec = args.batch_size * args.num_batches_per_iter / time
    log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
    img_secs.append(img_sec)

# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
log('Img/sec per %s: %.3f +-%.3f' % (device, img_sec_mean, img_sec_conf))
log('Total img/sec on %d %s(s): %.1f +-%.1f' %
    (args.world_size, device, args.world_size * img_sec_mean, args.world_size * img_sec_conf))
