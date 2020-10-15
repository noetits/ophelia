
from configuration import load_config
from data_load import *
import numpy as np
from synthesize import *
from synthesize_with_latent_space import compute_opensmile_features, load_features, load_embeddings
from architectures import *
from tqdm import tqdm
import random
import pandas as pd
# conf_file='/home/noetits/doctorat_code/ophelia/config/blizzard_letters.cfg'
conf_file='./config/blizzard_unsupervised_letters.cfg'
# conf_file='/home/noetits/doctorat_code/ophelia/config/will_unsupervised_letters_unsup_graph_old_preprocess.cfg'

hp=load_config(conf_file)
model_type='unsup'
logdir = hp.logdir + "-" + model_type 


##########################################

dataset=load_data(hp, mode='validation')


tts=tts_model(hp, model_type=model_type)

# test synthesis

fpaths, text_lengths, texts = dataset['fpaths'], dataset['text_lengths'], dataset['texts']
label_lengths, audio_lengths = dataset['label_lengths'], dataset['audio_lengths'] ## might be []

fnames = [os.path.basename(fpath) for fpath in fpaths]

ids=[fname.split('.')[0] for fname in fnames]

transcript=pd.read_csv(hp.transcript, sep='|', header=None)
transcript.index=transcript.iloc[:,0]
transcript=transcript.iloc[:,1]

for id in tqdm(transcript[transcript.index.str.contains(hp.validpatt)].index):
    if model_type=='unsup':
        melfile="{}/{}".format(hp.coarse_audio_dir, id+ ".npy")
        mel=np.load(melfile)
        mels=np.array([mel])
    else:
        mels=None
    tts.synthesize(text=transcript[id], mels=mels, id='val_'+id)


melfiles = ["{}/{}".format(hp.coarse_audio_dir, fname.replace("wav", "npy")) for fname in fnames]
texts=pd.read_csv('harvard_sentences.txt')[:3]
for melfile in tqdm(melfiles[:10]):
    mel=np.load(melfile)
    mels=np.array([mel])
    for i,text in texts.iterrows():
        print(text[0])
        sent=text[0].split('. ')[-1]
        print(sent)
        id=str(i)+'_ref_'+os.path.basename(melfile).split('.')[0]
        tts.synthesize(text=sent, mels=mels, id=id)

#g = Text2MelGraph(hp, mode="synthesize"); print("Graph 1 (t2m) loaded")
#g = Graph_style_unsupervised(hp, mode="train", load_in_memory=False)
#sess=tf.Session()
#sess.run(tf.global_variables_initializer())
#var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

#code=extract_emo_code(hp, mels, g)

#print(code)

#import pdb
#pdb.set_trace()