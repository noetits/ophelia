#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=1

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

#time python ./train.py -c ./config/will_unsupervised_letters_unsup_graph_old_preprocess.cfg -m t2m
#time python ./train.py -c ./config/will_unsupervised_letters_r_1.cfg -m unsup
#CUDA_VISIBLE_DEVICES=1 time python ./train.py -c ./config/kevin_smile_laughs.cfg -m t2m
#time python ./train.py -c ./config/laughs_only.cfg -m t2m
#time python ./train.py -c ./config/will_neutral_happy_phones_laughs.cfg -m t2m
#time python ./train.py -c ./config/will_neutral_happy_phones_kevin_neutral_smile_laughs.cfg -m ssrn
# CUDA_VISIBLE_DEVICES=1 time python ./train.py -c ./config/C.cfg -m t2m
# CUDA_VISIBLE_DEVICES=1 time python ./train.py -c ./config/kevin_smile_laughs_shortAEI.cfg -m t2m
# CUDA_VISIBLE_DEVICES=1 time python ./train.py -c ./config/laughs_shortAEI.cfg -m t2m
# CUDA_VISIBLE_DEVICES=1 time python ./train.py -c ./config/blizzard_letters.cfg -m t2m
CUDA_VISIBLE_DEVICES=0 time python ./train.py -c ./config/blizzard_unsupervised_letters.cfg -m unsup
