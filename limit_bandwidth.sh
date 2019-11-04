#!/bin/bash

sudo tc qdisc del dev enp5s0f0 root tbf rate ${1}Gbit latency ${2}ms burst ${3}kbit
sudo tc qdisc add dev enp5s0f0 root tbf rate ${1}Gbit latency ${2}ms burst ${3}kbit
