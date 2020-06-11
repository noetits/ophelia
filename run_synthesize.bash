#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2

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


#time python ./synthesize.py -c ./config/will_neutral_happy_phones_kevin_neutral_smile_laughs.cfg
#time python ./synthesize.py -c ./config/will_neutral_happy_phones_laughs.cfg
#time python ./synthesize.py -c ./config/lj_tutorial.cfg

# time python synthesize.py -c ./config/laughs.cfg #-babble
#time python synthesize.py -c ./config/kevin_smile_laughs_shortAEI.cfg 
time python synthesize.py -c ./config/laughs_shortAEI.cfg 