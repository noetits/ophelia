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
from architectures import Text2MelGraph, SSRNGraph, Graph_style_unsupervised
from synthesize import make_mel_batch, split_batch, synth_mel2mag
from configuration import load_config

import logger_setup
from logging import info

from data_load import *
import numpy as np
from synthesize import *

import pickle

def compute_opensmile_features(hp, conf_path='./tools/opensmile-2.3.0/config/gemaps/eGeMAPSv01a.conf', audio_extension='.wav', mode='train'):
    conf_name=conf_path.split('/')[-1].split('.')[0]
    dataset=load_data(hp, audio_extension=audio_extension, mode=mode)
    data=dataset['fpaths']
    dfs=[]
    for di, d in tqdm(enumerate(data)):
        print(str(di) + ' out of ' + str(len(data)))

        id_sentence=os.path.basename(d).split('.')[0]
        wave_path=d

        feature_path=os.path.join(hp.featuredir,'opensmile_features',conf_name)

        if (not os.path.exists(feature_path)):
            os.makedirs(feature_path)

        features_file=os.path.join(feature_path,id_sentence+'.csv')

        # opensmile only supports wave files (in 16 bit PCM), so if it is not (e.g. flac), we use librosa to load audio file and write temp.wav
        if wave_path.split('.')[-1]!='wav':
            y, sr = librosa.load(wave_path, sr=None)
            wave_path='temp.wav'
            sf.write(wave_path, y, sr, subtype='PCM_16')
        
        if not os.path.isfile(features_file): # if the file doesn't exist, compute features with opensmile
            opensmile_binary_path='./tools/opensmile-2.3.0/bin/linux_x64_standalone_static/'
            command = opensmile_binary_path+"SMILExtract -I {input_file} -C {conf_file} --csvoutput {output_file}".format(
                input_file=wave_path,
                conf_file=conf_path,
                output_file=features_file)
            os.system(command)
        #import pdb;pdb.set_trace()
        dfs.append(pd.read_csv(features_file, sep=';').iloc[0].iloc[2:]) # discard two first useless elements (name and frametime)
    feat_df=pd.concat(dfs, axis=1).transpose()
    feat_df.to_csv(os.path.join(feature_path,'feat_df.csv'))

def gather_opensmile_features(hp, conf_path='./tools/opensmile-2.3.0/config/gemaps/eGeMAPSv01a.conf', audio_extension='.wav', mode='train'):
    conf_name=conf_path.split('/')[-1].split('.')[0]
    dataset=load_data(hp, audio_extension=audio_extension, mode=mode)
    data=dataset['fpaths']
    dfs=[]
    for di, d in tqdm(enumerate(data)):
        #print(str(di) + ' out of ' + str(len(data)))
        id_sentence=os.path.basename(d).split('.')[0]
        wave_path=d
        feature_path=os.path.join(hp.featuredir,'opensmile_features',conf_name)
        features_file=os.path.join(feature_path,id_sentence+'.csv')
        
        dfs.append(pd.read_csv(features_file, sep=';').iloc[0].iloc[2:]) # discard two first useless elements (name and frametime)
    feat_df=pd.concat(dfs, axis=1).transpose()
    feat_df.to_csv(os.path.join(feature_path,'feat_df_'+mode+'.csv'))


def mi_regression_feat_embed(X, feat_df):
    '''
    X corresponds to latent embeddings
    feat_df is y, i.e. the acoustic features

    We want to see how much the acoustic features are predictable from the latent embeddings to
    check that they contain information about expressiveness.
    '''
    from sklearn.feature_selection import mutual_info_regression
    y=feat_df.values
    mi_embed=np.zeros((y.shape[-1],X.shape[-1]))
    #import pdb;pdb.set_trace()
    for idx in range(y.shape[-1]):
        mi_embed[idx,:]=mutual_info_regression(X, y[:,idx])
    
    mi_embed=pd.DataFrame(mi_embed)
    mi_embed.index=feat_df.columns
    
    return mi_embed



def regression_feat_embed(X, feat_df):
    from sklearn.linear_model import LinearRegression

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

        #corrs_embeds.append(corrs_embed)
        #coeffs.append(coeff)
    #corrs_embeds=np.array(corrs_embeds).T
    #coeffs = np.array(coeffs)

    corrs_embed_df=pd.DataFrame(corrs_embed)
    corrs_embed_df.index=feat_df.columns

    coeff_df = pd.DataFrame(coeff)
    coeff_df.index = feat_df.columns
    return corrs_embed_df, coeff_df


def corr_feat_embed(embed_dfs, feat_df, titles=[]):
    '''
    This function computes correlations between a set of features and each dimension of the embeddings.

    '''

    # rc('ytick', labelsize=8) #change text size

    n_feat = feat_df.shape[-1]
    corr_embeds=[]
    mi_embeds=[]
    for i,embed_df in enumerate(embed_dfs):
        embed_size = embed_df.shape[-1]

        ### Correlation matrix ###
        # feat_embed = pd.concat([feat_df, embed_df], axis=1)
        feat_embed=feat_df.copy()
        for i in range(embed_df.shape[-1]):
            feat_embed[str(i)]=embed_df.iloc[:,i]
        corr=feat_embed.astype(float).corr().abs()

        # # mi_embed=np.zeros((n_feat,embed_size))
        # # for dim in range(embed_size):
        # #     mi = mutual_info_regression(feat_df, embed_df[dim])
        # #     mi_embed[:,dim]=mi
        # # mi_embed=pd.DataFrame(mi_embed)
        # # mi_embed.index=feat_df.columns

        ### get one matrix for corr, F and mi with vad
        corr_embed=corr.iloc[:-embed_size,-embed_size:].abs()

        for i in range(embed_size):
            print('max corr '+str(i)+' : '+str(np.max(corr_embed.iloc[:,i])))

        corr_embeds.append(corr_embed)
        # mi_embeds.append(mi_embed)


    return corr_embeds, mi_embeds


def load_features(hp, conf_path='./tools/opensmile-2.3.0/config/gemaps/eGeMAPSv01a.conf'):
    import glob
    conf_name=conf_path.split('/')[-1].split('.')[0]
    feature_path=os.path.join(hp.featuredir,'opensmile_features',conf_name)
    paths=glob.glob(feature_path+'/*')
    if paths==[]:
        sys.exit('There is no feature file')
    dfs=[]
    for path in tqdm(paths):
        dfs.append(pd.read_csv(path, sep=';').iloc[0].iloc[2:]) # discard two first useless elements (name and frametime)
    feat_df=pd.concat(dfs, axis=1).transpose()
    #feat_df.index = pd.read_csv(path, sep=';').iloc[0].iloc[2:].index
    return feat_df

def get_emo_cats(hp):
    if hp.data_info:
        data_info=pd.read_csv(hp.data_info)
        dataset=load_data(hp)
        fpaths, text_lengths, texts = dataset['fpaths'], dataset['text_lengths'], dataset['texts']

        fnames = [os.path.basename(fpath) for fpath in fpaths]
        emo_cats=[data_info[data_info.id==fname.split('.')[0]]['emotion'].values[0] for fname in fnames]
        return emo_cats
    else:
        return None

def compute_unsupervised_embeddings(hp, g, model_type, mode='train'):
    dataset=load_data(hp, mode=mode)
    fpaths, text_lengths, texts = dataset['fpaths'], dataset['text_lengths'], dataset['texts']
    label_lengths, audio_lengths = dataset['label_lengths'], dataset['audio_lengths'] ## might be []

    fnames = [os.path.basename(fpath) for fpath in fpaths]
    melfiles = ["{}/{}".format(hp.coarse_audio_dir, fname.replace("wav", "npy")) for fname in fnames]
    codes=extract_emo_code(hp, melfiles, g, model_type)

    return codes

def save_embeddings(codes, logdir, filename='emo_codes', mode='train'):
    np.save(os.path.join(logdir,filename+'_'+mode+'.npy'),codes)

def load_embeddings(logdir, filename='emo_codes', mode='train'):
    codes=np.load(os.path.join(logdir,filename+'_'+mode+'.npy'))
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

def scatter_plot(matrice, c=None, s=1):
    import matplotlib.pyplot as plt
    import matplotlib
    #matplotlib.use('TkAgg')
    scatter=plt.scatter(matrice[:,0], matrice[:,1], c=c, s=s)
    return scatter


def main_work():
    
    # ============= Process command line ============
    a = ArgumentParser()
    a.add_argument('-c', dest='config', required=True, type=str)
    a.add_argument('-m', dest='model_type', required=True, choices=['t2m', 'unsup'])
    a.add_argument('-t', dest='task', required=True, choices=['compute_gradients','compute_codes', 'reduce_codes', 'compute_opensmile_features', 'show_plot','ICE_TTS','ICE_TTS_server'])
    a.add_argument('-r', dest='reduction_method', required=False, choices=['pca', 'tsne', 'umap'])
    a.add_argument('-p', dest='port', required=False, type=int, default=5000)
    opts = a.parse_args()
    print('opts')
    print(opts)
    # ===============================================
    model_type = opts.model_type
    method=opts.reduction_method
    hp = load_config(opts.config)
    logdir = hp.logdir + "-" + model_type 
    port=opts.port


    mode='validation'
    logger_setup.logger_setup(logdir)
    info('Command line: %s'%(" ".join(sys.argv)))
    print(logdir)
    task=opts.task
    if task=='compute_codes':
        if model_type=='t2m':
            g = Text2MelGraph(hp, mode="synthesize"); print("Graph 1 (t2m) loaded")
        elif model_type=='unsup':
            g = Graph_style_unsupervised(hp, mode="synthesize"); print("Graph 1 (unsup) loaded")
        codes=compute_unsupervised_embeddings(hp, g, model_type, mode=mode)
        save_embeddings(codes, logdir, mode=mode)
        #emo_cats=get_emo_cats(hp)
        #save(emo_cats, logdir, filename='emo_cats')
    elif task=='reduce_codes':
        try:
            embed=load_embeddings(logdir, mode=mode)[:,0,:]
        except IndexError: # I may have changed the shape of the matrix ...
            embed=load_embeddings(logdir, mode=mode)
        #import pdb;pdb.set_trace()
        model, results=embeddings_reduction(embed, method=method)
        save_embeddings(results, logdir, filename='emo_codes_'+method, mode=mode)
        save(model, logdir, filename='code_reduction_model_'+method)
    elif task=='compute_opensmile_features':
        compute_opensmile_features(hp, audio_extension='.wav', mode=mode)
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
        try:
            embed=load_embeddings(logdir, mode=mode)[:,0,:]
        except IndexError: # I may have changed the shape of the matrix ...
            embed=load_embeddings(logdir, mode=mode)
        
        print('Loading embeddings')
        embed_reduc=load_embeddings(logdir, filename='emo_codes_'+method)
        print('Loading emo cats')
        emo_cats=get_emo_cats(hp)
        #emo_cats=load(logdir, filename='emo_cats')
        #import pdb;pdb.set_trace()
        ice=ICE_TTS_server(hp, embed_reduc, embed, emo_cats, model_type=model_type, port=port)
        #ice=ICE_TTS_server(hp, embed_reduc, embed, model_type=model_type)
        #ice=ICE_TTS_server(hp, embed_reduc, embed, n_polar_axes=4, model_type=model_type)
    
    elif task=='compute_gradients':
        import seaborn as sns
        print('MODE', mode)
        try:
            embed=load_embeddings(logdir, mode=mode)[:,0,:]
        except IndexError: # I may have changed the shape of the matrix ...
            embed=load_embeddings(logdir, mode=mode)

        conf_name='eGeMAPSv01a'
        feature_path=os.path.join(hp.featuredir,'opensmile_features',conf_name,'feat_df_'+mode+'.csv') 
        feat_df=pd.read_csv(feature_path)
        feat_df=feat_df.drop(columns=['Unnamed: 0'])

        corrs_embed_df, coeff_df = regression_feat_embed(pd.DataFrame(embed), feat_df)
        print('Correlations:')
        #print(corrs_embed_df)
        # print('Gradients:')
        # print(coeff_df)
        # corrs_heatmap=sns.heatmap(feat_df.corr())
        # corrs_heatmap.get_figure().savefig('corrs_heatmap.png')

        print(corrs_embed_df.sort_values(0)[::-1][:20])

        #method='pca'

        embed_reduc=load_embeddings(logdir, filename='emo_codes_'+method, mode=mode)

        corrs_embed_reduc_df, coeff_reduc_df = regression_feat_embed(pd.DataFrame(embed_reduc), feat_df)
        print('Correlations:')
        #print(corrs_embed_reduc_df)
        #print('Gradients:')
        #print(coeff_reduc_df)

        print(corrs_embed_reduc_df.sort_values(0)[::-1][:20])

        #sc=scatter_plot(embed_reduc, c=feat_df['F0semitoneFrom27.5Hz_sma3nz_amean'].values)
        #sc.get_figure().savefig('scatter_'+method+'.png')

        mi=mi_regression_feat_embed(pd.DataFrame(embed_reduc), feat_df)

        print('mi',mi.sort_values(0)[::-1][:20])
        print('mi',mi.sort_values(1)[::-1][:20])

    else:
        print('Wrong task, does not exist')



if __name__=="__main__":

    main_work()
