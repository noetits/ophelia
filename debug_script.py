from configuration import load_config
from data_load import *
import numpy as np
from synthesize import *
from synthesize_with_latent_space import compute_opensmile_features, load_features, load_embeddings
from architectures import *
import soundfile as sf

conf_file='/home/noetits/doctorat_code/ophelia/config/blizzard_unsupervised_letters.cfg'
hp=load_config(conf_file)
model_type='unsup'
logdir = hp.logdir + "-" + model_type 
method='pca'

def synthesize_speech_laugh(hp):
    text1='this is just your ima'
    text2='imagination'
    tts=tts_model(hp)
    Y,Z,alignments=tts.synthesize(text=text1)
    Y2,Z2,alignments=tts.synthesize(text=text2)

##########################################

# test opensmile
conf_path='./tools/opensmile-2.3.0/config/gemaps/eGeMAPSv01a.conf'
conf_name=conf_path.split('/')[-1].split('.')[0]
#text_to_phonetic(text='this is a text')
dataset=load_data(hp, mode='demo', audio_extension='.flac')
#dataset=load_data(hp)
data=dataset['fpaths']
feature_path=os.path.join(logdir,'opensmile_features',conf_name)
fpaths=glob.glob(feature_path+'/*')

wave_path='/media/memory/noe/databases/WILL_FULL/wav_trim/enu_excla_0002.flac'
y, sr = librosa.load(wave_path, sr=None)

# test synthesis
synthesize(hp, 'Say hello to my little friend')
tts=tts_model(hp)
tts.synthesize(text='Say hello to my little friend')

fpaths, text_lengths, texts = dataset['fpaths'], dataset['text_lengths'], dataset['texts']
label_lengths, audio_lengths = dataset['label_lengths'], dataset['audio_lengths'] ## might be []

fnames = [os.path.basename(fpath) for fpath in fpaths]
melfiles = ["{}/{}".format(hp.coarse_audio_dir, fname.replace("wav", "npy")) for fname in fnames]

melfile=melfiles[0]
melfile='/home/noetits/doctorat_code/ophelia/work/will_unsupervised_1000/data/mels/enu_loud_prosoreq_0468.npy'
mel=np.load(melfile)

# upsample for copy synthesis from mel 2 wav with inverse mel filters and griffin lim
from scipy.signal import resample                                                                                                                                                        
melr=resample(mel, mel.shape[0]*4, axis=0)  
wavr=reconstruct_waveform(hp, melr) 
sf.write('test_GL/laugh_r_new_GL.wav', wavr, 22050)

magfile='/home/noetits/doctorat_code/ophelia/work/will_neutral_happy_phones_laughs/data/mags/valid_a_laugh6_2.npy'
mag=np.load(magfile)
wav=spectrogram2wav(hp,mag)
sf.write('test_GL/laugh_mag_GL.wav', wav, 22050)

mels=np.array([mel])

tts.synthesize(text='Say hello to my little friend', mels=mels)

g = Text2MelGraph(hp, mode="synthesize"); print("Graph 1 (t2m) loaded")
g = Graph_style_unsupervised(hp, mode="train", load_in_memory=False)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

code=extract_emo_code(hp, mels, g)

print(code)

import pdb
pdb.set_trace()