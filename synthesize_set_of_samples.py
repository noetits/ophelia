
from configuration import load_config
from data_load import *
import numpy as np
from synthesize import *
from synthesize_with_latent_space import load
from tqdm import tqdm
import pandas as pd
from itertools import product

# conf_file='/home/noetits/doctorat_code/ophelia/config/blizzard_letters.cfg'
conf_file='./config/blizzard_unsupervised_letters.cfg'
# conf_file='/home/noetits/doctorat_code/ophelia/config/will_unsupervised_letters_unsup_graph_old_preprocess.cfg'

hp=load_config(conf_file)
model_type='unsup'
logdir = hp.logdir + "-" + model_type 


##########################################

train_codes_pca=np.load(logdir+'/emo_codes_pca_train.npy')
min_xy=train_codes_pca.min(axis=0)
max_xy=train_codes_pca.max(axis=0)

pca_model=load(logdir)
a=np.mgrid[min_xy[0]:max_xy[0]:100j]
b=np.mgrid[min_xy[1]:max_xy[1]:100j]
X=np.array(list(product(a, b)))
codes=pca_model.inverse_transform(X)


def synthesize_set(hp, X, codes, inference_batch=1000):
    tts=tts_model(hp, model_type=model_type)
    # melfiles = ["{}/{}".format(hp.coarse_audio_dir, fname.replace("wav", "npy")) for fname in fnames]
    texts=pd.read_csv('harvard_sentences.txt')[:5]

    texts=[el.split('. ')[-1] for el in texts.iloc[:,0].tolist()]
    idxs=np.arange(len(X)).tolist()
    idxs = [str(i) for i in idxs] 

    for j,text in enumerate(texts):
        ids=['sent_'+str(j)+'_code_'+s for s in idxs]
        # tts.synthesize(text=[text]*len(X), emo_code=np.expand_dims(codes, axis=1), id=ids)

        for b in range(int(len(X)/inference_batch)):
            tts.synthesize(text=[text]*inference_batch, emo_code=np.expand_dims(codes[b*inference_batch:(b+1)*inference_batch], axis=1), id=ids[b*inference_batch:(b+1)*inference_batch])
        rest_idx=len(X)-len(X)%inference_batch
        tts.synthesize(text=[text]*(len(X)%inference_batch), emo_code=np.expand_dims(codes[rest_idx:], axis=1), id=ids[rest_idx:])

    # for i in tqdm(range(len(X))):
    #     print('code no. ', i)
    #     code = np.array([np.array([codes[i]])])
    #     for j,text in texts.iterrows():
    #         print(text[0])
    #         sent=text[0].split('. ')[-1]
    #         print(sent)
    #         # code_str='_'.join(str(list(X[i]))[1:-1].split(', '))
    #         id='sent_'+str(j)+'_code_'+str(i)
    #         tts.synthesize(text=sent, emo_code=code, id=id)

synthesize_set(hp, X, codes, inference_batch=3000)

# from server.ice_tts_server import ICE_TTS_server

# ice=ICE_TTS_server(hp, X, codes, web_page='web_page_play_presynthesized_samples.html')


# from architectures import *

#g = Text2MelGraph(hp, mode="synthesize"); print("Graph 1 (t2m) loaded")
#g = Graph_style_unsupervised(hp, mode="train", load_in_memory=False)
#sess=tf.Session()
#sess.run(tf.global_variables_initializer())
#var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

#code=extract_emo_code(hp, mels, g)

#print(code)

#import pdb
#pdb.set_trace()