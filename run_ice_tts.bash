#!/bin/bash

source activate py_ophelia_dctts

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3

echo "DATE : $(date)"
echo "_____________________________________________"
echo " HOSTNAME             : $HOSTNAME"
echo "_____________________________________________"
echo " CUDA_DEVICE_ORDER    : $CUDA_DEVICE_ORDER"
echo "_____________________________________________"
echo " CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "_____________________________________________"
nvidia-smi -L
echo "_____________________________________________"

time python ./synthesize_with_latent_space.py -c ./config/will_unsupervised.cfg -m t2m -t ICE_TTS_server -r umap

