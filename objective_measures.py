
'''
TODO: logSpecDbDist appropriate? (both mels & mags?)
TODO: compute output length error?
TODO: work out best way of handling the fact that predicted *coarse* features 
      can correspond to text but be arbitrarily 'out of phase' with reference.
      Mutliple references? Or compare against full-time resolution reference? 
'''
import logging
from mcd import dtw
import mcd.metrics_fast as mt

import numpy as np
import os

from configuration import load_config
conf_file='/home/noetits/doctorat_code/ophelia/config/blizzard_letters.cfg'
# conf_file='/home/noetits/doctorat_code/ophelia/config/blizzard_unsupervised_letters.cfg'
# conf_file='/home/noetits/doctorat_code/ophelia/config/will_unsupervised_letters_unsup_graph_old_preprocess.cfg'

hp=load_config(conf_file)
model_type='t2m'
logdir = hp.logdir + "-" + model_type 

import pyworld as pw
import soundfile as sf
import pandas as pd
import pysptk
from tqdm import tqdm
import pickle 

def compute_dtw_error(references, predictions, distance=mt.logSpecDbDist):
    minCostTot = 0.0
    framesTot = 0
    for (nat, synth) in tqdm(zip(references, predictions)):
        nat, synth = nat.astype('float64'), synth.astype('float64')
        minCost, path = dtw.dtw(nat, synth, distance)
        frames = len(nat)
        minCostTot += minCost
        framesTot += frames
    mean_score = minCostTot / framesTot
    print ('overall score = %f (%s frames nat/synth)' % (mean_score, framesTot))
    return mean_score

def get_dtw_aligned_predictions(references, predictions, distance=mt.logSpecDbDist):
    
    minCostTot = 0.0
    framesTot = 0
    for (nat, synth) in tqdm(zip(references, predictions)):
        nat, synth = nat.astype('float64'), synth.astype('float64')
        minCost, path = dtw.dtw(nat, synth, distance)
        frames = len(nat)
        minCostTot += minCost
        framesTot += frames
    mean_score = minCostTot / framesTot
    print ('overall score = %f (%s frames nat/synth)' % (mean_score, framesTot))
    return mean_score

def min_shifted_error(nat, synth, distance=mt.logSpecDbDist):
    mincost=float('Inf')
    for i in range(len(synth)):
        shifted_synth=np.concatenate([synth[i:,:], synth[:i,:]])

        cost = sum([
            distance(natFrame, synthFrame)
            for natFrame, synthFrame in zip(nat, shifted_synth)
        ])
        if cost<mincost: mincost=cost
    
    return mincost

def error_list(reference_list, prediction_list, distance=mt.logSpecDbDist):
    costs=[]
    frames=[]
    for (synth, nat) in tqdm(zip(prediction_list, reference_list)):
        #synth = prediction_tensor[i,:,:].astype('float64')
        # len_nat = len(nat)
        assert len(synth) == len(nat)
        #synth = synth[:len_nat, :]
        nat = nat.astype('float64')
        synth = synth.astype('float64')

        # cost = sum([
        #     distance(natFrame, synthFrame)
        #     for natFrame, synthFrame in zip(nat, synth)
        # ])

        cost=min_shifted_error(nat, synth, distance=distance)

        frames.append(len(nat))
        costs.append(cost)
    return costs, frames

# without DTW but with a translation
def compute_error(reference_list, prediction_list, distance=mt.logSpecDbDist):
    costs, frames = error_list(reference_list, prediction_list, distance=distance)
    return sum(costs)/sum(frames)

# def compute_error(reference_list, prediction_list, distance=mt.logSpecDbDist):
#     costTot = 0.0
#     framesTot = 0
#     for (synth, nat) in tqdm(zip(prediction_list, reference_list)):
#         #synth = prediction_tensor[i,:,:].astype('float64')
#         # len_nat = len(nat)
#         assert len(synth) == len(nat)
#         #synth = synth[:len_nat, :]
#         nat = nat.astype('float64')
#         synth = synth.astype('float64')
#         cost = sum([
#             distance(natFrame, synthFrame)
#             for natFrame, synthFrame in zip(nat, synth)
#         ])
#         framesTot += len(nat)
#         costTot += cost
#     return costTot / framesTot

def mgc_lf0_vuv(f0, sp, ap, fs=22050, order=13, alpha=None):
    if alpha is None:
        alpha=pysptk.util.mcepalpha(fs)
    #  https://github.com/r9y9/gantts/blob/master/prepare_features_tts.py
    mgc = pysptk.sp2mc(sp, order=order, alpha=alpha)
    # f0 = f0[:, None]
    lf0 = f0.copy()
    nonzero_indices = np.nonzero(f0)
    lf0[nonzero_indices] = np.log(f0[nonzero_indices])
    vuv = (ap[:, 0] < 0.5).astype(np.float32)[:, None]

    return mgc, lf0[:, None], vuv


transcript=pd.read_csv(hp.transcript, sep='|', header=None)
transcript.index=transcript.iloc[:,0]
transcript=transcript.iloc[:,1]

def compute_features_from_path(path):
    from tqdm import tqdm
    d={}
    d['sp_list']=[]
    d['f0_list']=[]
    d['ap_list']=[]
    for id in tqdm(transcript[transcript.index.str.contains(hp.validpatt)].index):
        file = [s for s in os.listdir(path) if id in s][0]
        wav,fs=sf.read(path+file)
        f0, sp, ap = pw.wav2world(wav, fs)
        # mgc, lf0, vuv = mgc_lf0_vuv(f0, sp, ap, fs=fs)
        d['sp_list'].append(sp)
        d['f0_list'].append(f0)
        d['ap_list'].append(ap)
    return d

def pad_feature(ref, pred, padding=1e-16):
    if len(ref)>len(pred):
        pad_size=len(ref)-len(pred)
        if len(ref.shape)>1:
            pred=np.concatenate((pred, padding*np.ones((pad_size, ref.shape[1]))))
        else:
            pred=np.concatenate((pred, padding*np.ones(pad_size)))
    elif len(pred)>len(ref):
        pad_size=len(pred)-len(ref)
        if len(ref.shape)>1:
            ref=np.concatenate((ref, padding*np.ones((pad_size, ref.shape[1]))))
        else:
            ref=np.concatenate((ref, padding*np.ones(pad_size)))
    return ref, pred

def pad_features(ref_list, pred_list, padding=1e-16):
    pad_ref_list=[]
    pad_pred_list=[]
    for i,ref in enumerate(ref_list):
        pred=pred_list[i]
        ref, pred=pad_feature(ref, pred, padding=1e-16)
        pad_ref_list.append(ref)
        pad_pred_list.append(pred)
    return pad_ref_list, pad_pred_list


def compute_and_save_features():
    ref_path='databases/ICE_TTS/blizzard2013/wavs/' 
    pred1_path='work/blizzard_letters/synth/t2m4000_ssrn14/reconstruction/'
    pred2_path='work/blizzard_unsupervised_letters/synth/t2mmodel_gs_504k_ssrn14/reconstruction/'
    # pred2_path='work/blizzard_unsupervised_letters/synth/t2m3942_ssrn14/reconstruction/'

    l=pred2_path.split('/')
    m=list(filter(None, l)) # drop empty elements ''
    t2m=m[-2]

    ref_d = compute_features_from_path(ref_path)
    pickle.dump(ref_d, open('feats_for_eval/ref_d.p','wb'))
    pred1_d = compute_features_from_path(pred1_path)
    pickle.dump(pred1_d, open('feats_for_eval/pred1_d.p','wb'))
    pred2_d = compute_features_from_path(pred2_path)
    pickle.dump(pred2_d, open('feats_for_eval/pred2_d_'+t2m+'.p','wb'))

    return ref_d,pred1_d,pred2_d

def load_features():
    ref_path='databases/ICE_TTS/blizzard2013/wavs/' 
    pred1_path='work/blizzard_letters/synth/t2m4000_ssrn14/reconstruction/'
    pred2_path='work/blizzard_unsupervised_letters/synth/t2mmodel_gs_504k_ssrn14/reconstruction/'
    # pred2_path='work/blizzard_unsupervised_letters/synth/t2m3942_ssrn14/reconstruction/'

    l=pred2_path.split('/')
    m=list(filter(None, l)) # drop empty elements ''
    t2m=m[-2]

    ref_d = pickle.load(open('feats_for_eval/ref_d.p','rb'))
    pred1_d = pickle.load(open('feats_for_eval/pred1_d.p','rb'))
    pred2_d = pickle.load(open('feats_for_eval/pred2_d_'+t2m+'.p','rb'))
    return ref_d, pred1_d, pred2_d


def compute_errors_from_lists(ref_d, pred_d, fs=22050):
    ref_mgc_list, pred_mgc_list, ref_vuv_list, pred_vuv_list, ref_lf0_list, pred_lf0_list = [],[],[],[],[],[]
    ref_mgc_list_pad, pred_mgc_list_pad, ref_vuv_list_pad, pred_vuv_list_pad, ref_lf0_list_pad, pred_lf0_list_pad = [],[],[],[],[],[]

    ref_f0_list, pred_f0_list = [],[]

    f0_continuous_list=[]
    

    for i in tqdm(range(len(ref_d['f0_list']))):
        # padding should correspond to silence: f0, sp, ap=pw.wav2world(np.zeros(50000),22050) 
        # import pdb;pdb.set_trace()
        ref_f0, pred_f0 = pad_feature(ref_d['f0_list'][i], pred_d['f0_list'][i], padding=0)
        ref_sp, pred_sp = pad_feature(ref_d['sp_list'][i], pred_d['sp_list'][i], padding=1e-16)
        ref_ap, pred_ap = pad_feature(ref_d['ap_list'][i], pred_d['ap_list'][i], padding=1)

        ref_mgc, ref_lf0, ref_vuv = mgc_lf0_vuv(ref_f0, ref_sp, ref_ap, fs=fs)
        pred_mgc, pred_lf0, pred_vuv = mgc_lf0_vuv(pred_f0, pred_sp, pred_ap, fs=fs)
        
        ref_mgc_list.append(ref_mgc[:,1:])
        ref_lf0_list.append(ref_lf0)
        ref_f0_list.append(ref_f0[:, None])
        ref_vuv_list.append(ref_vuv)

        pred_mgc_list.append(pred_mgc[:,1:])
        pred_lf0_list.append(pred_lf0)
        pred_f0_list.append(pred_f0[:, None])
        pred_vuv_list.append(pred_vuv)

    print('With DTW')
    MCD_dtw = compute_dtw_error(ref_mgc_list, pred_mgc_list)
    print('MCD', MCD_dtw)
    VDE_dtw = compute_dtw_error(ref_vuv_list, pred_vuv_list, mt.eucCepDist)
    print('VDE', VDE_dtw)
    lf0_MSE_dtw = compute_dtw_error(ref_lf0_list, pred_lf0_list, mt.sqCepDist)
    print('lf0_MSE', lf0_MSE_dtw)
    f0_MSE_dtw = compute_dtw_error(ref_f0_list, pred_f0_list, mt.sqCepDist)
    print('f0_MSE', f0_MSE_dtw)

    print('With shift')
    MCD_shift = compute_error(ref_mgc_list, pred_mgc_list)
    print('MCD', MCD_shift)
    VDE_shift = compute_error(ref_vuv_list, pred_vuv_list, mt.eucCepDist)
    print('VDE', VDE_shift)
    lf0_MSE_shift = compute_error(ref_lf0_list, pred_lf0_list, mt.sqCepDist)
    print('lf0_MSE', lf0_MSE_shift)
    f0_MSE_shift = compute_error(ref_f0_list, pred_f0_list, mt.sqCepDist)
    print('f0_MSE', f0_MSE_shift)

    dtw_errors=np.array([MCD_dtw, VDE_dtw, lf0_MSE_dtw, f0_MSE_dtw])
    print('dtw_errors', dtw_errors)
    shift_errors=np.array([MCD_shift, VDE_shift, lf0_MSE_shift, f0_MSE_shift])
    print('shift_errors', shift_errors)
    
    errors_df=pd.DataFrame([dtw_errors,shift_errors])
    errors_df.columns=['MCD','VDE','lf0 MSE','f0 MSE']
    errors_df.index=['DTW','shift']

    print(errors_df)
    print(errors_df.to_latex())

    # return MCD, VDE, F0_MSE

ref_d, pred1_d, pred2_d =  load_features()
print('-------  Errors between ref and classic TTS -------------')
compute_errors_from_lists(ref_d, pred1_d)
print('-------  Errors between ref and Unsup TTS -------------')
compute_errors_from_lists(ref_d, pred2_d)
