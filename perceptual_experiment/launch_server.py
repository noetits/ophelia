from ice_tts_server import ICE_TTS_server
import pickle
import numpy as np
import os
from itertools import product



# train_codes_pca=np.load('emo_codes_pca_train.npy')

# pca_model=pickle.load(open('code_reduction_model_pca.pkl', 'rb'))
# min_xy=train_codes_pca.min(axis=0)
# max_xy=train_codes_pca.max(axis=0)
# a=np.mgrid[min_xy[0]:max_xy[0]:100j]
# b=np.mgrid[min_xy[1]:max_xy[1]:100j]
# X=np.array(list(product(a, b)))
# codes=pca_model.inverse_transform(X)


X=np.load('X.npy')
codes=np.load('codes.npy')

import codecs, json 
X_json=json.dumps(X.tolist())
codes_json=json.dumps(codes.tolist())

json.dump(X_json, codecs.open('all_with_pca_limits/X.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format
json.dump(codes_json, codecs.open('all_with_pca_limits/codes.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format



ice=ICE_TTS_server(X, codes, web_page='web_page_play_presynthesized_samples_js_only.html')
