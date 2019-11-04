from configuration import load_config
from data_load import *
import numpy as np
from synthesize import *

conf_file='/home/noetits/doctorat_code/ophelia/config/will_unsupervised.cfg'
hp=load_config(conf_file)

#text_to_phonetic(text='this is a text')
#dataset=load_data(hp, mode='demo')
dataset=load_data(hp)

synthesize(hp, 'Say hello to my little friend')

tts=tts_model(hp)
tts.synthesize(hp, 'Say hello to my little friend')

fpaths, text_lengths, texts = dataset['fpaths'], dataset['text_lengths'], dataset['texts']
label_lengths, audio_lengths = dataset['label_lengths'], dataset['audio_lengths'] ## might be []

fnames = [os.path.basename(fpath) for fpath in fpaths]
melfiles = ["{}/{}".format(hp.coarse_audio_dir, fname.replace("wav", "npy")) for fname in fnames]

melfile=melfiles[0]
mel=np.load(melfile)
mels=np.array([mel])

g = Text2MelGraph(hp, mode="synthesize"); print("Graph 1 (t2m) loaded")

code=extract_emo_code(hp, mels, g)

print(code)

import pdb
pdb.set_trace()