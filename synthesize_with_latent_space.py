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
import soundfile

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

def compute_unsupervised_embeddings(hp):
    dataset=load_data(hp)
    fpaths, text_lengths, texts = dataset['fpaths'], dataset['text_lengths'], dataset['texts']
    label_lengths, audio_lengths = dataset['label_lengths'], dataset['audio_lengths'] ## might be []

    fnames = [os.path.basename(fpath) for fpath in fpaths]
    melfiles = ["{}/{}".format(hp.coarse_audio_dir, fname.replace("wav", "npy")) for fname in fnames]

    
    g = Text2MelGraph(hp, mode="synthesize"); print("Graph 1 (t2m) loaded")
    codes=extract_emo_code(hp, melfiles, g)

    return codes

def save_embeddings(codes, logdir, filename='emo_codes'):
    np.save(os.path.join(logdir,filename+'.npy'),codes)

def load_embeddings(logdir, filename='emo_codes'):
    codes=np.load(os.path.join(logdir,filename+'.npy'))
    return codes

def save_model(model, logdir, filename='code_reduction_model_pca'):
    pickle.dump(model, open(os.path.join(logdir,filename+'.pkl', 'wb')))

def load_model(logdir, filename='code_reduction_model_pca'):
    model = pickle.load(open(filename+'.pkl', 'rb'))
    return model

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
    a.add_argument('-t', dest='task', required=True, choices=['compute_codes', 'reduce_codes', 'show_plot','ICE-TTS'])
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
        codes=compute_unsupervised_embeddings(hp)
        save_embeddings(codes, logdir)
    elif task=='reduce_codes':
        embed=load_embeddings(logdir)[:,0,:]
        #import pdb;pdb.set_trace()
        model, results=embeddings_reduction(embed, method=method)
        save_embeddings(results, logdir, filename='emo_codes_'+method)
        save_model(model, logdir, filename='code_reduction_model_'+method)
    elif task=='show_plot':
        embed=load_embeddings(logdir, filename='emo_codes_'+method)
        scatter_plot(embed)
    elif task=='ICE-TTS':
        from interface import ICE_TTS
        embed=load_embeddings(logdir)[:,0,:]
        embed_reduc=load_embeddings(logdir, filename='emo_codes_'+method)
        from PyQt5.QtWidgets import QApplication
        app = QApplication(sys.argv)
        ice=ICE_TTS(hp, embed_reduc, embed)
        ice.show()
        sys.exit(app.exec_())
    else:
        print('Wrong task, does not exist')



if __name__=="__main__":

    main_work()
