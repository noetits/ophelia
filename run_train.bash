#!/bin/bash


source activate py_ophelia_dctts

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

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

time python ./train.py -c ./config/emov_db_unsupervised.cfg -m t2m

