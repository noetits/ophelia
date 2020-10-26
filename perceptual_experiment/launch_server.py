from ice_tts_server import ICE_TTS_server
import pickle
import numpy as np
import os
from itertools import product


logdir='/home/noetits/noe/work/blizzard_unsupervised_letters/train-unsup'
pca_model=pickle.load(open(os.path.join(logdir,'code_reduction_model_pca.pkl'), 'rb'))

train_codes_pca=np.load(logdir+'/emo_codes_pca_train.npy')
min_xy=train_codes_pca.min(axis=0)
max_xy=train_codes_pca.max(axis=0)

a=np.mgrid[min_xy[0]:max_xy[0]:100j]
b=np.mgrid[min_xy[1]:max_xy[1]:100j]
X=np.array(list(product(a, b)))
codes=pca_model.inverse_transform(X)

ice=ICE_TTS_server(X, codes, web_page='web_page_play_presynthesized_samples.html')
