#!/bin/bash

cd ~/distributed-training/test_scripts

mpirun -np 2 -H 172.31.24.165:1,172.31.18.73:1 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x PATH \
    -x NCCL_SOCKET_IFNAME=^lo,docker0,ens3 \
    -x PYTHONPATH \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include ens4 \
    python3 /home/ubuntu/distributed-training/test_scripts/pytorch_resnet50_cifar10.py \
    --epochs 1 --log-dir ../logs
