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
import logging

logging.getLogger('matplotlib.font_manager').disabled = True


from data_load import *
import numpy as np
from synthesize import *
import pickle
import matplotlib.pyplot as plt

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
    feat_df.to_csv(os.path.join(feature_path,'feat_df_'+mode+'.csv'))

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
    coeff_df = pd.DataFrame(coeff)
    coeff_df.index = feat_df.columns
    return reg, coeff_df

def test_regression(model, X, feat_df):
    y=feat_df.values
    y_pred=model.predict(X)

    corrs_embed=np.zeros(y.shape[-1])
    for idx in range(y.shape[-1]):
        corrs_embed[idx]=np.corrcoef([y_pred[:,idx],y[:,idx].astype(float)])[0,1]

    corrs_embed_df=pd.DataFrame(corrs_embed)
    corrs_embed_df.index=feat_df.columns
    return corrs_embed_df


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

def select_features(corrs_embed_df, feat_df, intra_corr_thresh=0.8, corr_thresh=0.3):
    intra_feat_corrs=feat_df.corr()
    selected_indices=[]
    sorted_corrs=corrs_embed_df.sort_values(0)[::-1]
    for i in range(len(sorted_corrs)):
        row=sorted_corrs.iloc[i]
        #print(row.name)
        
        # we check the correlations of the current feature with previous features
        bigger=intra_feat_corrs[sorted_corrs.index].T.iloc[:i,:][row.name].abs()>intra_corr_thresh
        too_much_correlated_with_previous=bigger.sum()>0
        if not too_much_correlated_with_previous:
            selected_indices.append(i)
    
    selected=sorted_corrs.iloc[selected_indices]
    high_corrs=selected.abs()>corr_thresh
    selected_high_corrs=selected[high_corrs].dropna()
    return selected_high_corrs

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

def scatter_plot(matrice, c=None, s=20, alpha=1):
    import matplotlib.pyplot as plt
    import matplotlib
    #matplotlib.use('TkAgg')
    plt.cla()
    scatter=plt.scatter(matrice[:,0], matrice[:,1], c=c, s=s, alpha=alpha, vmin=c.mean()-3*c.std(), vmax=c.mean()+3*c.std())
    plt.colorbar()
    return scatter

def plot_gradients(coeff,corr, ax=plt.gca()):
    import matplotlib
    matplotlib.use('Agg')
    # V=coeff.values
    # ax=plt.gca()
    origin = [0,0] # origin point
    # origin = [0], [0] # origin point
    # q=ax.quiver(*origin, V[:,0], V[:,1])
    from adjustText import adjust_text
    texts=[]
    for i in range(len(corr)):
        # if (corr.loc[coeff.index[i]].round(2).iloc[0])>0.5:
            grad=coeff[coeff.index==corr.index[i]].values[0]
            x=[origin[0], grad[0]]
            y = [origin[1], grad[1]]
            ax.plot(x, y, lw=2)
            # ax.legend()
            # ax.annotate(coeff.index[i], xy=(grad[0], grad[1]), xycoords='data')
            feat_name=corr.index[i].replace('_sma3', ' ').replace('nz','').replace('_','').replace('amean','mean').replace('semitoneFrom27.5Hz','')
            texts.append(ax.text(grad[0], grad[1], feat_name+' '+str(corr.iloc[i].iloc[0].round(2)), fontsize=9))
    adjust_text(texts, force_text=0.05, autoalign='xy', arrowprops=dict(arrowstyle="->", color='b', lw=1))
    plt.show()

def add_margin(ax,x=0.05,y=0.05):
    # This will, by default, add 5% to the x and y margins. You 
    # can customise this using the x and y arguments when you call it.

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xmargin = (xlim[1]-xlim[0])*x
    ymargin = (ylim[1]-ylim[0])*y

    ax.set_xlim(xlim[0]-xmargin,xlim[1]+xmargin)
    ax.set_ylim(ylim[0]-ymargin,ylim[1]+ymargin)

def abbridge_column_names(df):
    feats=[]
    for i in range(len(df.columns)):
        feat_name=df.columns[i].replace('_sma3', ' ').replace('nz','').replace('_','').replace('amean','mean').replace('semitoneFrom27.5Hz','').replace('stddev','std').replace('Stddev','std')
        feats.append(feat_name)
    df.columns=feats
    return df

def main_work():
    
    # ============= Process command line ============
    a = ArgumentParser()
    a.add_argument('-c', dest='config', required=True, type=str)
    a.add_argument('-m', dest='model_type', required=True, choices=['t2m', 'unsup'])
    a.add_argument('-t', dest='task', required=True, choices=['acoustic_analysis','compute_codes', 'reduce_codes', 'compute_opensmile_features', 'show_plot','ICE_TTS','ICE_TTS_server'])
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
        embed_reduc=load_embeddings(logdir, filename='emo_codes_'+method, mode=mode)
        print('Loading emo cats')
        emo_cats=get_emo_cats(hp)
        #emo_cats=load(logdir, filename='emo_cats')
        #import pdb;pdb.set_trace()
        ice=ICE_TTS_server(hp, embed_reduc, embed, emo_cats, model_type=model_type, port=port)
        #ice=ICE_TTS_server(hp, embed_reduc, embed, model_type=model_type)
        #ice=ICE_TTS_server(hp, embed_reduc, embed, n_polar_axes=4, model_type=model_type)
    
    elif task=='acoustic_analysis':
        import seaborn as sns
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_regression
        from sklearn.linear_model import LinearRegression
        from pandas.plotting import scatter_matrix
        # from pandas.plotting._matplotlib.misc import scatter_matrix
        import matplotlib.pyplot as plt 
        from scipy.stats import pearsonr
        import itertools

        print('MODE', mode)
        try:
            embed=load_embeddings(logdir, mode=mode)[:,0,:]
            embed_valid=load_embeddings(logdir, mode='validation')[:,0,:]
        except IndexError: # I may have changed the shape of the matrix ...
            embed=load_embeddings(logdir, mode=mode)
            embed_valid=load_embeddings(logdir, mode='validation')

        conf_name='eGeMAPSv01a'
        feature_path=os.path.join(hp.featuredir,'opensmile_features',conf_name,'feat_df_'+mode+'.csv') 
        feat_df=pd.read_csv(feature_path)
        feat_df=feat_df.drop(columns=['Unnamed: 0'])

    
        feature_path=os.path.join(hp.featuredir,'opensmile_features',conf_name,'feat_df_'+'validation'+'.csv') 
        feat_df_valid=pd.read_csv(feature_path)
        #import pdb;pdb.set_trace()
        feat_df_valid=feat_df_valid.drop(columns=['Unnamed: 0'])

        feat_df=abbridge_column_names(feat_df)
        feat_df_valid=abbridge_column_names(feat_df_valid)

        # Mean normalization (with same mean and variance computed from training data)
        feat_df=(feat_df-feat_df.mean())/feat_df.std() 
        feat_df_valid=(feat_df_valid-feat_df.mean())/feat_df.std() 

        model, coeff_df = regression_feat_embed(pd.DataFrame(embed), feat_df)
        corrs_embed_df=test_regression(model, pd.DataFrame(embed_valid), feat_df_valid)
        print('Correlations:')
        print(corrs_embed_df.sort_values(0)[::-1][:20])

        selected=select_features(corrs_embed_df, feat_df_valid, intra_corr_thresh=0.7, corr_thresh=0.3)
        print(selected.to_latex().replace('\_sma3', ' ').replace('nz','').replace('\_','').replace('amean','mean').replace('semitoneFrom27.5Hz',''))
        # print('Gradients:')
        # print(coeff_df)
        

        #method='pca'

        embed_reduc=load_embeddings(logdir, filename='emo_codes_'+method, mode=mode)
        embed_reduc_valid=load_embeddings(logdir, filename='emo_codes_'+method, mode='validation')

        model_reduc, coeff_reduc_df = regression_feat_embed(pd.DataFrame(embed_reduc), feat_df)
        corrs_embed_reduc_df=test_regression(model_reduc, pd.DataFrame(embed_reduc_valid), feat_df_valid)
        print('Correlations:')
        print(corrs_embed_reduc_df.sort_values(0)[::-1][:20])

        selected_reduc=select_features(corrs_embed_reduc_df, feat_df_valid, intra_corr_thresh=0.7, corr_thresh=0.25)
        print(selected.to_latex().replace('\_sma3', ' ').replace('nz','').replace('\_','').replace('amean','mean').replace('semitoneFrom27.5Hz',''))

        feat_predictions_df=pd.DataFrame(model.predict(embed))
        feat_predictions_df.index=feat_df.index
        feat_predictions_df.columns=feat_df.columns

        feat_df[selected.index]
        feat_predictions_df[selected.index]

        # just checking it seems correct
        # print(pearsonr(feat_df[selected.index]['F0semitoneFrom27.5Hz_sma3nz_percentile50.0'],feat_predictions_df[selected.index]['F0semitoneFrom27.5Hz_sma3nz_percentile50.0'] ))
        
        # selected_feats=selected.index.to_list()
        # fig, axs = plt.subplots(nrows=sc.shape[0], ncols=sc.shape[1], figsize=(100, 100))
        # for pair in itertools.product(range(len(selected)), repeat=2): 
        #     x=feat_df[selected_feats[pair[0]]]
        #     y=feat_predictions_df[selected_feats[pair[1]]]
        #     axs[pair[0], pair[1]].scatter(x, y, alpha=0.2)
        # fig.savefig('figures/scatter_matrix.png')

        
        h=100
        selected_feats=selected.index.to_list()
        fig, axs = plt.subplots(nrows=len(selected), ncols=1, figsize=(h/len(selected)*3, h))
        for i in range(len(selected)):
            x=feat_df[selected_feats[i]]
            y=feat_predictions_df[selected_feats[i]]
            axs[i].scatter(x, y, alpha=0.2)
        fig.savefig('figures/scatter_plots_feats.png')

        
        #print(corrs_embed_reduc_df)
        print('Gradients:')
        print(coeff_reduc_df)

        normalized_gradients=coeff_reduc_df.div(((coeff_reduc_df**2).sum(axis=1))**0.5, axis=0)
        
        plt.cla()
        plt.clf() 
        plt.close()
        # sc=scatter_plot(embed_reduc, c=feat_df['F0semitoneFrom27.5Hz_sma3nz_amean'].values)
        sc=scatter_plot(embed_reduc, c=feat_df['F0 mean'].values)
        plot_gradients(normalized_gradients,selected_reduc, ax=sc.get_figure().gca())
        sc.get_figure().savefig('figures/scatter_F0_mean_'+method+'.png')


        plt.cla()
        plt.clf() 
        plt.close()
        # sc=scatter_plot(embed_reduc, c=feat_df['F0semitoneFrom27.5Hz_sma3nz_amean'].values)
        sc=scatter_plot(embed_reduc, c=feat_df['F0 percentile50.0'].values)
        plot_gradients(normalized_gradients,selected_reduc, ax=sc.get_figure().gca())
        sc.get_figure().savefig('figures/scatter_F0_percentile50.0_'+method+'.png')

        print(feat_df.columns)
        # import pdb;pdb.set_trace()
        plt.cla()
        plt.clf() 
        plt.close()
        # sc=scatter_plot(embed_reduc, c=feat_df['F0semitoneFrom27.5Hz_sma3nz_amean'].values)
        sc=scatter_plot(embed_reduc, c=feat_df['F3amplitudeLogRelF0 stdNorm'].values)
        plot_gradients(normalized_gradients,selected_reduc, ax=sc.get_figure().gca())
        sc.get_figure().savefig('figures/scatter_F3amplitudeLogRelF0_stdNorm_'+method+'.png')

        
        plt.cla()
        plt.clf() 
        plt.close()
        # sc=scatter_plot(embed_reduc, c=feat_df['F0semitoneFrom27.5Hz_sma3nz_amean'].values)
        sc=scatter_plot(embed_reduc, c=feat_df['stdVoicedSegmentLengthSec'].values)
        plot_gradients(normalized_gradients,selected_reduc, ax=sc.get_figure().gca())
        sc.get_figure().savefig('figures/scatter_stdVoicedSegmentLengthSec_'+method+'.png')

        
        plt.cla()
        plt.clf() 
        plt.close()
        hist=sns.distplot(feat_df['F0 mean'])
        hist.get_figure().savefig('figures/hist_F0_mean_'+method+'.png')

        # hist=sns.distplot(feat_df['F3amplitudeLogRelF0 stddevNorm'])
        # hist.get_figure().savefig('figures/hist_F3amplitudeLogRelF0_stddevNorm_'+method+'.png')



        #mi=mi_regression_feat_embed(pd.DataFrame(embed_reduc), feat_df)
        #print('mi',mi.sort_values(0)[::-1][:20])
        #print('mi',mi.sort_values(1)[::-1][:20])

        # Plot corrs heatmaps
        plt.close()
        corrs_heatmap_feats=sns.heatmap(feat_df.corr().abs(), xticklabels=False)
        corrs_heatmap_feats.get_figure().savefig('figures/corrs_heatmap_feats.pdf', bbox_inches='tight')

        plt.close()
        embed_corr=pd.DataFrame(embed).corr().abs()
        embed_corr_heatmap=sns.heatmap(embed_corr)
        embed_corr_heatmap.get_figure().savefig('figures/embed_corr_heatmap.pdf', bbox_inches='tight')

        plt.close()
        corr_feat_embed=pd.concat([pd.DataFrame(embed),feat_df], axis=1).corr().abs()
        sns.set(font_scale=0.2)
        corr_feat_embed_heatmap=sns.heatmap(corr_feat_embed, xticklabels=False)
        # add_margin(corr_feat_embed_heatmap,x=0.1,y=0.0)
        corr_feat_embed_heatmap.get_figure().savefig('figures/corr_feat_embed_heatmap.pdf', bbox_inches='tight')
        

    else:
        print('Wrong task, does not exist')



if __name__=="__main__":

    main_work()
