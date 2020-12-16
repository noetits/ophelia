from ice_tts_server import ICE_TTS_server
from ice_tts_server import PlotResource
import pickle
import numpy as np
import os
from itertools import product

from matplotlib.patches import Rectangle


def idx_to_case_xy(idx):
    x=int(int(idx/100)/100*5)
    y=int((idx-100*int(idx/100))/100*5)
    return x,y


# train_codes_pca=np.load('emo_codes_pca_train.npy')

# pca_model=pickle.load(open('code_reduction_model_pca.pkl', 'rb'))
# min_xy=train_codes_pca.min(axis=0)
# max_xy=train_codes_pca.max(axis=0)
# a=np.mgrid[min_xy[0]:max_xy[0]:100j]
# b=np.mgrid[min_xy[1]:max_xy[1]:100j]
# X=np.array(list(product(a, b)))
# codes=pca_model.inverse_transform(X)


X=np.load('X.npy')
codes=np.load('codes.npy')

plotRes=PlotResource(X, codes)


# assuming that we have 10000 samples (100x100), see "synthesize_set_of_samples.py" at root

n=100
n_cases=5
step=n/n_cases
x=np.arange(0,n,step)

rows=x+step/2
columns_hundreads=(rows)*100

idxs=[]
for c in columns_hundreads:
    idxs+=(c+rows).astype(int).tolist()
    print(c+rows)

a=np.arange(n_cases)
cases=np.array(list(product(a, a)))

def map_range(x, x0, x1, y0, y1):
    '''
    Map the number n from the range x0,x1 to the range y0,y1
    '''
    print('x : {}, x0 : {}, x1 : {} '.format(x,x0,x1))
    print('y0 : {}, y1 : {} '.format(y0,y1))
    #import pdb;pdb.set_trace()
    nRel=(x-x0)/(x1-x0)
    return nRel*(y1-y0)+y0




matrice=X[idxs]
plotRes.ax.scatter(matrice[:,0], matrice[:,1], c='red', edgecolors='none')
plotRes.ax.grid(False)
plotRes.ax.get_xaxis().set_ticks([])
plotRes.ax.get_yaxis().set_ticks([])

plotRes.fig.savefig('plot_grid.png')


w=(X[:,0].max()-X[:,0].min())/n_cases
h=(X[:,1].max()-X[:,1].min())/n_cases

for el in cases:
    plotRes.ax.patches = []
    x=(map_range(el[0], 0, n_cases, X[:,0].min(), X[:,0].max()))
    y=(map_range(el[1], 0, n_cases, X[:,1].min(), X[:,1].max()))
    rect = Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
    plotRes.ax.add_patch(rect)
    plotRes.fig.savefig('plot_grid_'+str(el[0]) + str(el[1]) +'.png')
