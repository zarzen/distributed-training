from __future__ import print_function

import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import models
import horovod.torch as hvd
import timeit
import numpy as np

# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')

parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=50,
                    help='number of benchmark iterations')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

hvd.init()

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())

cudnn.benchmark = True

# Set up standard model.
model = getattr(models, args.model)()

if args.cuda:
    # Move model to GPU.
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters(),
                                     compression=compression)

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Set up fixed fake data
data = torch.randn(args.batch_size, 3, 224, 224)
target = torch.LongTensor(args.batch_size).random_() % 1000
if args.cuda:
    data, target = data.cuda(), target.cuda()

step1 = torch.cuda.Event(enable_timing=True)
step2 = torch.cuda.Event(enable_timing=True)
step3 = torch.cuda.Event(enable_timing=True)
step4 = torch.cuda.Event(enable_timing=True)
step5 = torch.cuda.Event(enable_timing=True)
step6 = torch.cuda.Event(enable_timing=True)

time_zero_grad = []
time_model = []
time_loss = []
time_back = []
time_step = []
time_batch = []

def benchmark_step():
    step1.record()
    optimizer.zero_grad()
    step2.record()
    output = model(data)
    step3.record()
    loss = F.cross_entropy(output, target)
    step4.record()
    loss.backward()
    step5.record()
    optimizer.step()
    step6.record()
    torch.cuda.synchronize()
    time_zero_grad.append(step2.elapsed_time(step1))
    time_model.append(step3.elapsed_time(step2))
    time_loss.append(step4.elapsed_time(step3))
    time_back.append(step5.elapsed_time(step4))
    time_step.append(step6.elapsed_time(step5))
    time_batch.append(step6.elapsed_time(step1))



def log(s, nl=True):
    if hvd.rank() != 0:
        return
    print(s, end='\n' if nl else '')


log('Model: %s' % args.model)
log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, hvd.size()))

# Warm-up
log('Running warmup...')
timeit.timeit(benchmark_step, number=args.num_warmup_batches)

# Benchmark
log('Running benchmark...')
img_secs = []
for x in range(args.num_iters):
    time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
    img_sec = args.batch_size * args.num_batches_per_iter / time
    if x % 20 == 0:
        log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
    img_secs.append(img_sec)

# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
log('Img/sec per %s: %.3f +-%.3f' % (device, img_sec_mean, img_sec_conf))
log('Total img/sec on %d %s(s): %.1f +-%.1f' %
    (hvd.size(), device, hvd.size() * img_sec_mean, hvd.size() * img_sec_conf))
if hvd.rank() == 0:
    print("zero_grad", np.mean(time_zero_grad))
    print("model", np.mean(time_model))
    print("loss", np.mean(time_loss))
    print("back", np.mean(time_back))
    print("step", np.mean(time_step))
    print("batch", np.mean(time_batch))
