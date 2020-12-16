import numpy as np
import random
random.seed(5)
import pandas as pd
# assuming that we have 10000 samples (100x100), see "synthesize_set_of_samples.py" at root

n=100
step=n/5
x=np.arange(0,n,step)

rows=x+step/2
columns_hundreads=(rows)*100

idxs=[]
for c in columns_hundreads:
    idxs+=(c+rows).astype(int).tolist()
    print(c+rows)

turkle_inputs=["ref_sent_id","ref_idx","latent_sent_id"]

# random.shuffle(idxs)
# print(idxs)

# ref_sent_ids=[]
latent_sent_ids=[]
for i in range(5):
    # ref_sent_ids+=[i]*5
    # latent_sent_ids+=[i]*5
    latent_sent_ids+=[i]*25

# random.shuffle(ref_sent_ids)

ref_sent_ids=np.random.randint(4, size=len(latent_sent_ids))
ref_sent_ids=[idx if idx<latent_sent_ids[i] else idx+1 for i,idx in enumerate(ref_sent_ids)]

# df = pd.DataFrame(list(zip(ref_idxs, idxs, latent_sent_ids)), 
#                columns =turkle_inputs) 

df = pd.DataFrame(list(zip(latent_sent_ids, idxs*5, latent_sent_ids)), 
               columns =turkle_inputs) 
df.sample(frac=1, random_state=1).to_csv('batch_same_text_many.csv', index=None)

df = pd.DataFrame(list(zip(latent_sent_ids, idxs*5, ref_sent_ids)), 
               columns =turkle_inputs) 
df.sample(frac=1, random_state=1).to_csv('batch_diff_text_many.csv', index=None)


# df.to_csv('batch_same_text_many.csv', index=None)