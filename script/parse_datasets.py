import numpy as np
import codecs
import os
# import unicodedata
import pandas as pd
import csv
import tqdm
import soundfile as sf

####### The following functions come from: #########
# https://github.com/sterling239/audio-emotion-recognition

def get_all_files(path_to_wav):
    files = os.listdir(path_to_wav)
    return files

def get_emotions(path_to_emotions, filename):
    # type: (object, object) -> object
    f = open(path_to_emotions + filename, 'r').read()
    # print f
    # print f.split('\n')
    f = np.array(f.split('\n'))  # np.append(np.array(['']), np.array(f.split('\n')))
    c = 0
    idx = f == ''
    idx_n = np.arange(len(f))[idx]
    emotion = []
    for i in range(len(idx_n) - 2):
        g = f[idx_n[i] + 1:idx_n[i + 1]]
        head = g[0]
        i0 = head.find(' - ')
        start_time = float(head[head.find('[') + 1:head.find(' - ')])
        end_time = float(head[head.find(' - ') + 3:head.find(']')])
        actor_id = head[
                   head.find(filename[:-4]) + len(filename[:-4]) + 1:head.find(filename[:-4]) + len(filename[:-4]) + 5]
        emo = head[head.find('\t[') - 3:head.find('\t[')]
        vad = head[head.find('\t[') + 1:]

        v = float(vad[1:7])
        a = float(vad[9:15])
        d = float(vad[17:23])

        emotion.append({#'start': start_time,
                        #'end': end_time,
                        'id': filename[:-4] + '_' + actor_id,
                        'v': v,
                        'a': a,
                        'd': d,
                        'emotion': emo})
    return emotion

# available_emotions = ['ang', 'exc', 'fru', 'neu', 'sad']

def get_transcriptions(path_to_transcriptions, filename):
    f = open(path_to_transcriptions + filename, 'r').read()
    f = np.array(f.split('\n'))
    transcription = {}

    for i in range(len(f) - 1):
        g = f[i]
        i1 = g.find(': ')
        i0 = g.find(' [')
        ind_id = g[:i0]
        ind_ts = g[i1 + 2:]
        transcription[ind_id] = ind_ts
    return transcription

import re
def load_iemocap(path_to_IEMOCAP):
    data = []
    sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
    for session in sessions:
        print(session)
        path_to_wav = os.path.join(path_to_IEMOCAP, session+ '/dialog/wav/')
        path_to_sentences = os.path.join(path_to_IEMOCAP, session+ '/sentences/wav/')
        path_to_emotions = os.path.join(path_to_IEMOCAP, session+ '/dialog/EmoEvaluation/')
        path_to_transcriptions = os.path.join(path_to_IEMOCAP, session+ '/dialog/transcriptions/')

        # files lists dialogs
        files = get_all_files(path_to_wav)
        files = np.array([x for x in files if not x.startswith('.')]) #remove files beginning by '.'
        files=files[[f[-4:] == '.wav' for f in files]]# keep only if ends by '.wav'
        files = [f[:-4] for f in files] # remove this .wav extension from the names
        print(len(files))
        print(files)

        for f in files:
            # f corresponds to one dialog
            emotions = get_emotions(path_to_emotions, f + '.txt')
            transcriptions = get_transcriptions(path_to_transcriptions, f + '.txt')
            for ie, e in enumerate(emotions):
                # It means that we discard files without a category assigned to it...
                # I'm not sure but I suppose it is when the opinions of annotators were not consistent
                # if not(e['emotion']=='xxx'):
                # e['left'] = sample[ie]['left']
                # e['right'] = sample[ie]['right']
                e['database']='IEMOCAP'
                # if '[' in transcriptions[e['id']]:
                #     import pdb;pdb.set_trace()

                transcription=re.sub(' [[A-Z]*] ', '', transcriptions[e['id']])
                transcription=re.sub('[[A-Z]*] ', '', transcription)
                transcription=re.sub(' [[A-Z]*]', '', transcription)
                transcription=re.sub('[[A-Z]*]', '', transcription)
                e['transcription'] = transcription
                e['sentence_path']=path_to_sentences+f+'/'+e['id']+'.wav'
                e['speaker']=session+'_'+e['id'][-4]
                data.append(e)

    # data = np.array(data)
    # pdb.set_trace()
    data=pd.DataFrame.from_records(data)

    # print(data)
    print(len(data))
    return data

######################################################

def load_acapela(path_to_ACAPELA, emo_cat):
    data = []
    speaker='WILL'
    category_folders=emo_cat #['WILL_HAPPY', 'WILL_NEUTRAL', 'WILL_SAD']
    # emotion_categories={'WILL_HAPPY':'hap','WILL_NEUTRAL':'neu','WILL_SAD':'sad'}
    for cat in category_folders:
        print(cat)
        path_to_wav = os.path.join(path_to_ACAPELA, cat)
        files = get_all_files(path_to_wav)
        files = np.array([x for x in files if not x.startswith('.')])  # remove files beginning by '.'
        files = files[[f[-4:] == '.wav' for f in files]]  # keep only if ends by '.wav'
        files = [f[:-4] for f in files]  # remove this .wav extension from the names
        print(len(files))
        # print(files)

        for f in files:
            # pdb.set_trace()
            dic=pd.read_csv(path_to_wav+'/'+f+'.dic', sep='[', quoting=csv.QUOTE_NONE)
            words=dic.iloc[:,1].tolist()
            words = [w[:-2] for w in words]
            transcription=' '.join(words)
            e={'database':'ACAPELA',
               'id':f,
               'speaker':speaker,
               'emotion':cat,
               'transcription':transcription,
               'sentence_path':path_to_wav+'/'+f+'.wav'}
            data.append(e)
            # pdb.set_trace()

    # data = np.array(data)
    # pdb.set_trace()
    data = pd.DataFrame.from_records(data)

    # print(data)
    print(len(data))
    return data


def parse_dic(path):
    dic=pd.read_csv(path, sep='[', quoting=csv.QUOTE_NONE)
    words=dic.iloc[:,1].tolist()
    words = [w[:-2] for w in words]
    transcription=' '.join(words)
    return transcription

import librosa
def parse_breath_group(dic_path, phn_path, sentence_path, sr=22050):
    phn=pd.read_csv(phn_path, header=None, sep=' ')
    dic=pd.read_csv(dic_path, sep='[', quoting=csv.QUOTE_NONE)
    dic_silences_indices=dic.loc[dic['_ ] ']=='_ ] '].index.tolist()
    phn_silences_indices=phn.loc[phn[2]=='_'].index.tolist()

    sample_start=phn_silences_indices[0]
    phn_silences_indices.pop(0)

    y,sr=librosa.load(sentence_path, sr=sr)
    transcriptions=[]
    wavs=[]
    start=-1
    for i_sil in range(len(dic_silences_indices)):
        sample_end=phn.iloc[phn_silences_indices[i_sil],0]

        end=dic_silences_indices[i_sil]
        #print(dic[start+1:end+1])

        words=dic[start+1:end+1].iloc[:,1].tolist()
        words = [w[:-2] for w in words]
        transcription=' '.join(words)
        transcriptions.append(transcription)
        wavs.append(y[sample_start:sample_end])

        start=end
        sample_start=phn.iloc[phn_silences_indices[i_sil],1]
    
    return transcriptions, wavs


def load_full_will(path_to_ACAPELA='databases/WILL_FULL', audio_extension='.flac'):
    data = []
    speaker='WILL'
    category_folders=next(os.walk(path_to_ACAPELA))[1]
    # category_folders=emo_cat #['WILL_HAPPY', 'WILL_NEUTRAL', 'WILL_SAD']
    # emotion_categories={'WILL_HAPPY':'hap','WILL_NEUTRAL':'neu','WILL_SAD':'sad'}
    for cat in category_folders:
        print(cat)
        path_to_wav = os.path.join(path_to_ACAPELA, cat, 'WAV22')
        path_to_text= os.path.join(path_to_ACAPELA, cat, 'SEG_NUU')
        files = get_all_files(path_to_wav)
        files = np.array([x for x in files if not x.startswith('.')])  # remove files beginning by '.'
        files = files[[f.split('.')[-1] == audio_extension.split('.')[-1] for f in files]]  # keep only if ends by '.wav'
        files = [f.split('.')[0] for f in files]  # remove this .wav extension from the names
        print(len(files))
        for f in files:
            # pdb.set_trace()
            transcription=parse_dic(path_to_text+'/'+f+'.dic')
            e={'database':'WILL_FULL',
               'id':f,
               'speaker':speaker,
               'emotion':cat,
               'transcription':transcription,
               'sentence_path':path_to_wav+'/'+f+audio_extension}
            data.append(e)
    data = pd.DataFrame.from_records(data)
    print(len(data))
    return data

def load_blizzard2013(path='databases/blizzard2013/train/segmented/', audio_extension='.wav'):
    data=[]
    files = get_all_files(os.path.join(path,'SEG_NUU'))
    files = np.array([x for x in files if not x.startswith('.')])  # remove files beginning by '.'
    files = files[[f.split('.')[-1] == 'dic' for f in files]]  # keep only if ends by '.dic'
    files = [f.split('.')[0] for f in files]  # remove this .wav extension from the names
    print(len(files))

    for f in tqdm.tqdm(files):
        # pdb.set_trace()
        wav_path=os.path.join(path,'wavn',f+audio_extension)
        dic_path=os.path.join(path,'SEG_NUU',f+'.dic')
        
        transcription=parse_dic(dic_path)
        e={'database':'blizzard2013',
            'id':f,
            #'speaker':speaker,
            #'emotion':cat,
            'transcription':transcription,
            'sentence_path':wav_path}
        data.append(e)
    data = pd.DataFrame.from_records(data)
    return data

def parse_blizzard2013_by_breath_group(path='databases/blizzard2013/train/segmented/', audio_extension='.wav', outpath='databases/ICE_TTS/blizzard2013/wavs16bits/', sr=22050):
    if not os.path.exists(outpath): os.makedirs(outpath) 

    df=load_blizzard2013(path=path, audio_extension=audio_extension)

    all_transcriptions={}
    all_segments=[]
    data=[]
    for i,row in tqdm.tqdm(df.iterrows()):
        dic_path=os.path.join(path,'SEG_NUU',row.id+'.dic')
        phn_path=os.path.join(path,'SEG_NUU',row.id+'.phn22')
        transcriptions, wavs=parse_breath_group(dic_path, phn_path, row.sentence_path, sr=sr)
        for i,wav in enumerate(wavs):
            e={
            'id':'_'.join([row.id,'seg',str(i)]),
            'transcription':transcriptions[i],
            }
            #print(e)
            data.append(e)
            all_transcriptions[e['id']]=transcriptions[i]
            #import pdb;pdb.set_trace()
            #maxv = np.iinfo(np.int16).max
            #librosa.output.write_wav(os.path.join(outpath,e['id']+audio_extension),(wav * maxv).astype(np.int16),sr)
            sf.write(os.path.join(outpath,e['id']+audio_extension), wav, sr, subtype='PCM_16')

            
    data = pd.DataFrame.from_records(data)
    #data_to_transcript()

    return data




def load_emov_db(path_to_EmoV_DB):
    print('Load EmoV-DB')
    # TODO: remove this line and load all speakers
    # emo_path = os.path.split(path_to_EmoV_DB)[0]
    transcript = os.path.join(path_to_EmoV_DB, 'cmuarctic.data')
    lines = codecs.open(transcript, 'r', 'utf-8').readlines()

    # in our database, we use only files beginning with arctic_a. And the number of these sentences correspond.
    # Here we build a dataframe with number and text of each of these lines
    sentences = []
    for line in lines:
        temp = {}
        idx_n_0 = line.find('arctic_a') + len('arctic_a')
        if line.find('arctic_a') != -1:
            # print(line)
            # print(idx_n_0)
            idx_n_end = idx_n_0 + 4
            number = line[idx_n_0:idx_n_end]
            # print(number)
            temp['n'] = number
            idx_text_0 = idx_n_end + 2
            text = line.strip()[idx_text_0:-3]
            temp['text'] = text
            # print(text)
            sentences.append(temp)
    sentences = pd.DataFrame(sentences)

    # speaker='bea'
    speakers=next(os.walk(path_to_EmoV_DB))[1]

    data=[]

    for speaker in speakers:
        print(speaker)
        emo_cat=next(os.walk(os.path.join(path_to_EmoV_DB,speaker)))[1]
        for emo in emo_cat:
            print(emo)
            if os.path.exists(os.path.join(path_to_EmoV_DB, speaker, emo)):
                for file in os.listdir(os.path.join(path_to_EmoV_DB, speaker, emo)):
                    #print(file)
                    fpath = os.path.join(path_to_EmoV_DB, speaker, emo, file)

                    if file[-4:] == '.wav':
                        fnumber = file[-8:-4]
                        # result must be a string and not a df with a single element.
                        try:
                            text = sentences[sentences['n'] == fnumber]['text'].iloc[0]
                        except:
                            print('In load_emov_db: did not find the sentence coresponding to '+str(file)+', check if the filename is standard (similar with others)')
                        e = {'database': 'EmoV-DB',
                             'id': file[:-4],
                             'speaker': speaker,
                             'emotion':emo,
                             'transcription': text,
                             'sentence_path': fpath}
                        data.append(e)
                        #print(e)
    print('All speakers loaded')
    data = pd.DataFrame.from_records(data)
    print('EmoV-DB dataframe built')
    return data

def load_LJ(path_to_LJ):
    transcript = os.path.join(path_to_LJ, 'metadata.csv')
    lines = codecs.open(transcript, 'r', 'utf-8').readlines()

    data=[]
    for line in lines:
        fname, _, text = line.strip().split("|")

        fpath = os.path.join(path_to_LJ, "wavs", fname + ".wav")

        e = {'database': 'LJ',
             'id': fname,
             'speaker': 'LJ_speaker',
             # 'emotion':emotion_categories[cat],
             'transcription': text,
             'sentence_path': fpath}
        data.append(e)

    data = pd.DataFrame.from_records(data)
    return data

import glob

# part of this function is from https://github.com/buriburisuri/speech-to-text-wavenet (apache 2.0)
def load_VCTK(path_to_VCTK):
    # read label-info
    df = pd.read_csv(os.path.join(path_to_VCTK, 'speaker-info.txt'), usecols=['ID'], index_col=False, delim_whitespace=True)

    # read file IDs
    file_ids = []
    for d in [path_to_VCTK + '/txt/p%d/' % uid for uid in df.ID.values]:
        file_ids.extend([f[-12:-4] for f in sorted(glob.glob(d + '*.txt'))])

    data=[]
    for i, f in enumerate(file_ids):

        # wave file name
        fpath = path_to_VCTK + '/wav48/%s/' % f[:4] + f + '.wav'

        # get label index
        text = open(path_to_VCTK + '/txt/%s/' % f[:4] + f + '.txt').read()

        e = {'database': 'VCTK',
             'id': f,
             'speaker': f[:4],
             # 'emotion':emotion_categories[cat],
             'transcription': text,
             'sentence_path': fpath}
        data.append(e)

    data = pd.DataFrame.from_records(data)
    return data

def load_libriTTS(path_to_libriTTS='databases/LibriTTS', category='train-clean-100'):
    full_path=os.path.join(path_to_libriTTS, category)
    data=[]
    # read directory list by speaker
    speaker_list = glob.glob(full_path + '/*')
    for spk in speaker_list:
        # read directory list by chapter
        chapter_list = glob.glob(spk + '/*/')
        for chap in chapter_list:

            # read label text file list
            file_list = glob.glob(chap + '/*.wav')
            for fpath in file_list:
                #print(fpath)
                txt=fpath[:-4]+'.normalized.txt'
                with open(txt, 'rt') as f:
                    text = f.read()
                    id=os.path.split(fpath)[1].split('.')[0]
                    #print(id)
                    e = {'database': 'LibriTTS',
                         'id': id,
                         'speaker': spk.split('/')[-1],
                         # 'emotion':emotion_categories[cat],
                         'transcription': text,
                         'sentence_path': fpath}
                    data.append(e)

    data = pd.DataFrame.from_records(data)
    return data

def data_to_transcript(data, outpath='databases/ICE_TTS/LibriTTS/transcript.csv'):
    d=data[['id','transcription','transcription']]
    d.to_csv(outpath, sep='|', index=False)