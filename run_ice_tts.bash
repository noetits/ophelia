#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=""

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

time python ./synthesize_with_latent_space.py -c ./config/blizzard_unsupervised_letters.cfg -m unsup -t ICE_TTS_server -r pca -p 5001
# time python ./synthesize_with_latent_space.py -c ./config/will_unsupervised_letters_unsup_graph_old_preprocess.cfg -m unsup -t reduce_codes -r pca -p 5001
# time python ./synthesize_with_latent_space.py -c ./config/will_unsupervised_letters_not_norm.cfg -m t2m -t ICE_TTS_server -r pca -p 5001

