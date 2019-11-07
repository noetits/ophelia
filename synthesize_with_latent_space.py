from utils import *
from data_load import load_data
import pandas as pd

import sys
import os
import glob
from argparse import ArgumentParser

import imp

import numpy as np

from utils import spectrogram2wav
# from scipy.io.wavfile import write
import soundfile as sf

import tqdm
from concurrent.futures import ProcessPoolExecutor

import tensorflow as tf
from architectures import Text2MelGraph, SSRNGraph
from synthesize import make_mel_batch, split_batch, synth_mel2mag
from configuration import load_config

import logger_setup
from logging import info

from data_load import *
import numpy as np
from synthesize import *

import pickle

def compute_opensmile_features(hp, logdir, conf_path='./tools/opensmile-2.3.0/config/gemaps/eGeMAPSv01a.conf', audio_extension='.wav'):
    conf_name=conf_path.split('/')[-1].split('.')[0]
    dataset=load_data(hp, audio_extension=audio_extension)
    data=dataset['fpaths']
    for di, d in tqdm(enumerate(data)):
        print(str(di) + ' out of ' + str(len(data)))

        id_sentence=os.path.basename(d).split('.')[0]
        wave_path=d

        feature_path=os.path.join(logdir,'opensmile_features',conf_name)
        #database_path = '/'.join(os.path.normpath(wave_path).split('/')[1:-1])

        if (not os.path.exists(feature_path)):
            os.makedirs(feature_path)

        features_file=os.path.join(feature_path,id_sentence+'.csv')

        # opensmile only supports wave files (in 16 bit PCM), so if it is not (e.g. flac), we use librosa to load audio file and write temp.wav
        if wave_path.split('.')[-1]!='wav':
            y, sr = librosa.load(wave_path, sr=None)
            # import pdb;pdb.set_trace()
            wave_path='temp.wav'
            sf.write(wave_path, y, sr, subtype='PCM_16')
            #maxv = np.iinfo(np.int16).max
            #y *= maxv / max(0.01, np.max(np.abs(y)))
            #librosa.output.write_wav(wave_path, y.astype(np.int16), sr)


        if not os.path.isfile(features_file): # if the file doesn't exist, compute features with opensmile
            opensmile_binary_path='./tools/opensmile-2.3.0/bin/linux_x64_standalone_static/'
            command = opensmile_binary_path+"SMILExtract -I {input_file} -C {conf_file} --csvoutput {output_file}".format(
                input_file=wave_path,
                conf_file=conf_path,
                output_file=features_file)
            os.system(command)


def regression_feat_embed(embed_dfs, feat_df):
    from sklearn.linear_model import LinearRegression

    corrs_embeds=[]
    coeffs=[]
    for i,embed_df in enumerate(embed_dfs):
        # embed_size = embed_df.shape[-1]

        X=embed_df.values
        y=feat_df.values
        reg = LinearRegression().fit(X, y)
        coeff=reg.coef_
        intercept=reg.intercept_
        print(coeff.shape)
        print(intercept.shape)
        y_pred=reg.predict(X)
        print(X.shape)
        print(y.shape)

        # Test linear regression manually
        # pred=np.dot(coeff,np.array([X[0,:]]).T)+np.array([intercept]).T
        # print(np.max(y_pred[0,:]-pred[:,0])) # This should be 0 (or almost)


        corrs_embed=np.zeros(y.shape[-1])
        for idx in range(y.shape[-1]):
            corrs_embed[idx]=np.corrcoef([y_pred[:,idx],y[:,idx].astype(float)])[0,1]

        corrs_embeds.append(corrs_embed)
        coeffs.append(coeff)
    corrs_embeds=np.array(corrs_embeds).T
    coeffs = np.array(coeffs)

    corrs_embeds_df=pd.DataFrame(corrs_embeds)
    corrs_embeds_df.index=feat_df.columns

    # coeffs_df = pd.DataFrame(coeffs)
    # coeffs_df.index = feat_df.columns
    return corrs_embeds_df, coeffs


def compute_unsupervised_embeddings(hp):
    dataset=load_data(hp)
    fpaths, text_lengths, texts = dataset['fpaths'], dataset['text_lengths'], dataset['texts']
    label_lengths, audio_lengths = dataset['label_lengths'], dataset['audio_lengths'] ## might be []

    fnames = [os.path.basename(fpath) for fpath in fpaths]
    melfiles = ["{}/{}".format(hp.coarse_audio_dir, fname.replace("wav", "npy")) for fname in fnames]

    data_info=pd.read_csv(hp.data_info)
    emo_cats=[data_info[data_info.id==fname.split('.')[0]]['emotion'].values[0] for fname in fnames]

    g = Text2MelGraph(hp, mode="synthesize"); print("Graph 1 (t2m) loaded")
    codes=extract_emo_code(hp, melfiles, g)

    return codes, emo_cats

def save_embeddings(codes, logdir, filename='emo_codes'):
    np.save(os.path.join(logdir,filename+'.npy'),codes)

def load_embeddings(logdir, filename='emo_codes'):
    codes=np.load(os.path.join(logdir,filename+'.npy'))
    return codes

def save(var, logdir, filename='code_reduction_model_pca'):
    pickle.dump(var, open(os.path.join(logdir,filename+'.pkl'), 'wb'))

def load(logdir, filename='code_reduction_model_pca'):
    var = pickle.load(open(os.path.join(logdir,filename+'.pkl'), 'rb'))
    return var

def embeddings_reduction(embed, method='pca'):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    print('Reducing with method '+method)
    
    if method == 'pca':
        model = PCA(n_components=2)
        results = model.fit_transform(embed)
    elif method == 'tsne':
        model = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        results = model.fit_transform(embed)
    elif method == 'umap':
        import umap
        model=umap.UMAP()
        results = model.fit_transform(embed)
    else:
        print('Wrong dimension reduction method')
    return model, results

def scatter_plot(matrice):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    plt.scatter(matrice[:,0], matrice[:,1])
    plt.show()


def main_work():
    
    # ============= Process command line ============
    a = ArgumentParser()
    a.add_argument('-c', dest='config', required=True, type=str)
    a.add_argument('-m', dest='model_type', required=True, choices=['t2m', 'ssrn', 'babbler'])
    a.add_argument('-t', dest='task', required=True, choices=['compute_codes', 'reduce_codes', 'compute_opensmile_features', 'show_plot','ICE_TTS','ICE_TTS_server'])
    a.add_argument('-r', dest='reduction_method', required=False, choices=['pca', 'tsne', 'umap'])
    opts = a.parse_args()
    print('opts')
    print(opts)
    # ===============================================
    model_type = opts.model_type
    method=opts.reduction_method
    hp = load_config(opts.config)
    logdir = hp.logdir + "-" + model_type 
    logger_setup.logger_setup(logdir)
    info('Command line: %s'%(" ".join(sys.argv)))
    print(logdir)
    task=opts.task
    if task=='compute_codes':
        codes, emo_cats=compute_unsupervised_embeddings(hp)
        save_embeddings(codes, logdir)
        save(emo_cats, logdir, filename='emo_cats')
    elif task=='reduce_codes':
        embed=load_embeddings(logdir)[:,0,:]
        #import pdb;pdb.set_trace()
        model, results=embeddings_reduction(embed, method=method)
        save_embeddings(results, logdir, filename='emo_codes_'+method)
        save(model, logdir, filename='code_reduction_model_'+method)
    elif task=='compute_opensmile_features':
        compute_opensmile_features(hp, logdir, audio_extension='.flac')
    elif task=='show_plot':
        embed=load_embeddings(logdir, filename='emo_codes_'+method)
        scatter_plot(embed)
    elif task=='ICE_TTS':
        from interface import ICE_TTS
        embed=load_embeddings(logdir)[:,0,:]
        embed_reduc=load_embeddings(logdir, filename='emo_codes_'+method)
        from PyQt5.QtWidgets import QApplication
        app = QApplication(sys.argv)
        ice=ICE_TTS(hp, embed_reduc, embed)
        ice.show()
        sys.exit(app.exec_())
    elif task=='ICE_TTS_server':
        from server.ice_tts_server import ICE_TTS_server
        embed=load_embeddings(logdir)[:,0,:]
        embed_reduc=load_embeddings(logdir, filename='emo_codes_'+method)
        emo_cats=load(logdir, filename='emo_cats')
        ice=ICE_TTS_server(hp, embed_reduc, embed, emo_cats)

    else:
        print('Wrong task, does not exist')



if __name__=="__main__":

    main_work()
