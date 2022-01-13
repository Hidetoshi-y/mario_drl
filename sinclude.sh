#!/bin/bash

# debug
set -xe
#export SLURM_DEBUG=1

# environment valiables
#export TF_FORCE_GPU_ALLOW_GROWTH=true
#export TF_CPP_MIN_LOG_LEVEL=3

# variables
IMAGE='nvcr.io#nvidia/pytorch:21.08-py3'
REPO=${REPO:-$(basename $PWD)}
GPUS=${GPUS:-'1'}
OPTS=${OPTS:-'--container-mounts=.:/workspace --container-workdir=/workspace'}
PROG=${PROG:-'--pty /bin/bash -i'}

# user hook
if [ x"$_included" != x"yes" ]; then
    _included=yes
    if [ -f sinclude.sh ]; then
	echo "./sinclude.sh found, use it."
	source ./sinclude.sh
    fi
fi
#pfrl用のsinclude.sh