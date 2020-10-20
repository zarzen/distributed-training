from __future__ import print_function

import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
# import tensorboardX
import os
import math
from tqdm import tqdm
from logger import get_logger, log_time, sync_e
import json
import time
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--train-dir', default=os.path.expanduser('/tmp/ramdisk'),
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

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=90,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=2,
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

torch.manual_seed(args.seed)

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(0)
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

# Horovod: print logs on the first worker.
verbose = 1

# Horovod: write TensorBoard logs on first worker.
# log_writer = tensorboardX.SummaryWriter(args.log_dir) if hvd.rank() == 0 else None
log_writer = None

model_logger = get_logger(0)

# Horovod: limit # of CPU threads to be used per worker.
torch.set_num_threads(4)

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
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
    train_dataset, num_replicas=1, rank=0)
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
    val_dataset, num_replicas=1, rank=0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                         sampler=val_sampler, **kwargs)


# Set up standard model.
model = getattr(models, args.model)()

if args.cuda:
    # Move model to GPU.
    model.cuda()

# Horovod: scale learning rate by the number of GPUs.
# Gradient Accumulation: scale learning rate by batches_per_allreduce
optimizer = optim.SGD(model.parameters(),
                      lr=(args.base_lr *
                          args.batches_per_allreduce),
                      momentum=args.momentum, weight_decay=args.wd)

# Horovod: (optional) compression algorithm.
# compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
# optimizer = hvd.DistributedOptimizer(
#     optimizer, named_parameters=model.named_parameters(),
#     compression=compression,
#     backward_passes_per_step=args.batches_per_allreduce)

# Restore from a previous checkpoint, if initial_epoch is specified.
# Horovod: restore on the first worker which will broadcast weights to other workers.
if resume_from_epoch > 0:
    filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Horovod: broadcast parameters & optimizer state.
# hvd.broadcast_parameters(model.state_dict(), root_rank=0)
# hvd.broadcast_optimizer_state(optimizer, root_rank=0)

step1 = torch.cuda.Event(enable_timing=True)
step2 = torch.cuda.Event(enable_timing=True)
step3 = torch.cuda.Event(enable_timing=True)
step4 = torch.cuda.Event(enable_timing=True)
step5 = torch.cuda.Event(enable_timing=True)
step6 = torch.cuda.Event(enable_timing=True)
step7 = torch.cuda.Event(enable_timing=True)
step8 = torch.cuda.Event(enable_timing=True)
step9 = torch.cuda.Event(enable_timing=True)
step10 = torch.cuda.Event(enable_timing=True)
step11 = torch.cuda.Event(enable_timing=True)
step12 = torch.cuda.Event(enable_timing=True)
step13 = torch.cuda.Event(enable_timing=True)
step14 = torch.cuda.Event(enable_timing=True)

time_load_data = []
time_zero_grad = []
time_model = []
time_accu = []
time_accu_update = []
time_loss = []
time_loss_update = []
time_back = []
time_step = []
time_batch = []

step1.record()

def train(epoch):
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')

    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            # number of batchs limit
            if batch_idx >= 500:
                return

            if args.cuda:
                with log_time(model_logger, "batch-data-tocuda", 0):
                    data, target = data.cuda(), target.cuda()
            step2.record()
            adjust_learning_rate(epoch, batch_idx)
            step3.record()
            optimizer.zero_grad()
            step4.record()
            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), args.batch_size):
                data_batch = data[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]
                # sync_e()
                lobj = {"ph": "X", "name": "foward", "ts": time.time(), "pid": 0, "dur": 0}
                output = model(data_batch)
                step5.record()
                # event record 
                # sync_e()
                lobj["dur"]=time.time()-lobj["ts"]
                model_logger.info(json.dumps(lobj))

                lobj = {"ph": "X", "name": "compute-loss", "ts": time.time(), "pid": 0, "dur": 0}
                step6.record()
                with log_time(model_logger, "horovod-acc-comp", 0):
                    _acc = accuracy(output, target_batch)
                step7.record()
                with log_time(model_logger, "horovod-acc-update", 0):
                    train_accuracy.update(_acc)
                step8.record()
                with log_time(model_logger, "torch-loss-comp", 0):
                    loss = F.cross_entropy(output, target_batch)
                step9.record()
                with log_time(model_logger, "horovod-loss-update", 0):
                    train_loss.update(loss)
                step10.record()
                # Average gradients among sub-batches
                with log_time(model_logger, "avg-sub-batches-loss", 0):
                    loss.div_(math.ceil(float(len(data)) / args.batch_size))
                lobj["dur"]=time.time()-lobj["ts"]
                model_logger.info(json.dumps(lobj))

                # sync_e()
                lobj = {"ph": "X", "name": "backward", "ts": time.time(), "pid": 0, "dur": 0}
                step11.record()
                loss.backward()
                step12.record()
                # sync_e()
                lobj["dur"]=time.time()-lobj["ts"]
                model_logger.info(json.dumps(lobj))

            # Gradient is applied across all ranks
            lobj = {"ph": "X", "name": "update-gradients", "ts": time.time(), "pid": 0, "dur": 0}
            step13.record()
            optimizer.step()
            step14.record()
            torch.cuda.synchronize()
            time_batch.append(step14.elapsed_time(step1))
            step1.record()
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
        lr_adj = 1.
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * args.batches_per_allreduce * lr_adj


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(epoch):
    filepath = args.checkpoint_format.format(epoch=epoch + 1)
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filepath)


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += val
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

lobj = {"ph": "X", "name": "training", "ts": time.time(), "pid": 0, "dur": 0}
for epoch in range(resume_from_epoch, args.epochs):
    train(epoch)
    # validate(epoch)
    # save_checkpoint(epoch)

lobj["dur"]=time.time()-lobj["ts"]
model_logger.info(json.dumps(lobj))
print("batch", np.mean(time_batch[40:]))
    