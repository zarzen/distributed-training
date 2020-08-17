from __future__ import print_function

import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms, models

import os
import math
from tqdm import tqdm
from logger import get_logger, log_time, sync_e
import json
import time

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

def train(epoch):
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')

    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            adjust_learning_rate(epoch, batch_idx)
            # number of batchs limit
            if batch_idx >= 50:
                return 
            with log_time(model_logger, "batch-data-tocuda", args.local_rank):
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), args.batch_size):
                data_batch = data[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]
                # sync_e()
                lobj = {"ph": "X", "name": "foward", "ts": time.time(), "pid": args.local_rank, "dur": 0}
                output = model(data_batch)
                # event record 
                # sync_e()
                lobj["dur"]=time.time()-lobj["ts"]
                model_logger.info(json.dumps(lobj))

                lobj = {"ph": "X", "name": "compute-loss", "ts": time.time(), "pid": args.local_rank, "dur": 0}
                with log_time(model_logger, "horovod-acc-comp", args.local_rank):
                    _acc = accuracy(output, target_batch)
                with log_time(model_logger, "horovod-acc-update", args.local_rank):
                    train_accuracy.update(_acc)
                with log_time(model_logger, "torch-loss-comp", args.local_rank):
                    loss = F.cross_entropy(output, target_batch)
                with log_time(model_logger, "horovod-loss-update", args.local_rank):
                    train_loss.update(loss)
                # Average gradients among sub-batches
                with log_time(model_logger, "avg-sub-batches-loss", args.local_rank):
                    loss.div_(math.ceil(float(len(data)) / args.batch_size))
                lobj["dur"]=time.time()-lobj["ts"]
                model_logger.info(json.dumps(lobj))

                # sync_e()
                lobj = {"ph": "X", "name": "backward", "ts": time.time(), "pid": args.local_rank, "dur": 0}
                loss.backward()
                # sync_e()
                lobj["dur"]=time.time()-lobj["ts"]
                model_logger.info(json.dumps(lobj))

            # Gradient is applied across all ranks
            lobj = {"ph": "X", "name": "update-gradients", "ts": time.time(), "pid": args.local_rank, "dur": 0}
            optimizer.step()
            lobj["dur"]=time.time()-lobj["ts"]
            model_logger.info(json.dumps(lobj))

            t.set_postfix({'loss': train_loss.avg.item(),
                           'accuracy': 100. * train_accuracy.avg.item()})
            t.update(1)

    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)


def validate(epoch):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)


# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / args.world_size * (epoch * (args.world_size - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * args.world_size * args.batches_per_allreduce * lr_adj


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(epoch):
    if args.world_id * args.num_gpu + args.local_rank == 0:
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)

def init(args):
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


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        new_sum = val.detach().cuda()
        dist.all_reduce(new_sum)
        self.sum += torch.div(new_sum, args.world_size)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='resnet50',
                        help='model to benchmark')
    parser.add_argument('--train-dir', default=os.path.expanduser('~/data/imagenet/train'),
                        help='path to training data')
    parser.add_argument('--val-dir', default=os.path.expanduser('~/data/imagenet/validation'),
                        help='path to validation data')
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
    parser.add_argument('--num-gpu', type=int, default=1,
                        help='num of gpu')

    # Default settings from https://arxiv.org/abs/1706.02677.
    parser.add_argument('--url-port', type=str, default='tcp://172.31.22.234:23456',
                        help='tcp://172.31.22.234:23456')
    parser.add_argument('--world-size', type=int, default=32,
                        help='world size')

    parser.add_argument('--world-id', type=int, default=0,
                        help='world id')

    parser.add_argument('--local_rank', type=int, default=0,
                        help='local id')

    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=32,
                        help='input batch size for validation')
    parser.add_argument('--epochs', type=int, default=90,
                        help='number of epochs to train')
    parser.add_argument('--base-lr', type=float, default=0.0125,
                        help='learning rate for a single GPU')
    parser.add_argument('--warmup-epochs', type=float, default=5,
                        help='number of warmup epochs')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--wd', type=float, default=0.00005,
                        help='weight decay')

    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')

    args = parser.parse_args()

    allreduce_batch_size = args.batch_size * args.batches_per_allreduce

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    init(args)

    cudnn.benchmark = True

    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            resume_from_epoch = try_epoch
            break


    # Establish Local Rank and set device on this node 设置节点的本地化编号和设备
    local_rank = int(args.local_rank)
    dp_device_ids = [local_rank]
    torch.cuda.set_device(local_rank)

    # Horovod: print logs on the first worker.
    verbose = 1 if args.world_id * args.num_gpu + args.local_rank == 0 else 0

    # Horovod: write TensorBoard logs on first worker.
    # log_writer = tensorboardX.SummaryWriter(args.log_dir) if hvd.rank() == 0 else None
    log_writer = None

    model_logger = get_logger(args.local_rank)

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(4)

    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_dataset = \
        datasets.ImageFolder(args.train_dir,
                            transform=transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                            ]))
    # Horovod: use DistributedSampler to partition data among workers. Manually specify
    # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=args.world_id * args.num_gpu + args.local_rank)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=allreduce_batch_size,
        sampler=train_sampler, **kwargs)

    val_dataset = \
        datasets.ImageFolder(args.val_dir,
                            transform=transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                            ]))
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=args.world_size, rank=args.world_id * args.num_gpu + args.local_rank)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                            sampler=val_sampler, **kwargs)


    # Set up standard model.
    model = getattr(models, args.model)().cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=dp_device_ids, output_device=local_rank)

    # Horovod: scale learning rate by the number of GPUs.
    # Gradient Accumulation: scale learning rate by batches_per_allreduce
    optimizer = optim.SGD(model.parameters(),
                        lr=(args.base_lr *
                            args.batches_per_allreduce * args.world_size),
                        momentum=args.momentum, weight_decay=args.wd)

    # Restore from a previous checkpoint, if initial_epoch is specified.
    # Horovod: restore on the first worker which will broadcast weights to other workers.
    if resume_from_epoch > 0 and args.world_id * args.num_gpu + args.local_rank == 0:
        filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    lobj = {"ph": "X", "name": "training", "ts": time.time(), "pid": args.local_rank, "dur": 0}
    for epoch in range(resume_from_epoch, args.epochs):
        train(epoch)
        # validate(epoch)
        # save_checkpoint(epoch)

    lobj["dur"]=time.time()-lobj["ts"]
    model_logger.info(json.dumps(lobj))
