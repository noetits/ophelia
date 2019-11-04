import falcon
import argparse
import numpy as np
import soundfile as sf
import scipy
import io

class UIResource:
    def on_get(self, req, res):
        res.content_type = 'text/html'
        with open ("server/web_page.html", "r") as myfile:
            html_body=myfile.read()
        res.body = html_body

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('pdf')
import json
from data_load import load_data
import pandas as pd
class PlotResource:
    def __init__(self,  hp, plot_data, codes, emo_cats=None):
        self.hp=hp
        self.plot_data=plot_data
        self.codes=codes
        self.scatter_plot(plot_data, emo_cats)
    def scatter_plot(self,matrice, emo_cats=None):
        self.fig, self.ax = plt.subplots()
        
        df=pd.DataFrame()
        df['style']=emo_cats
        for g in pd.unique(df['style']):
            i = np.where(df['style'] == g)
            print(i)
            self.ax.scatter(matrice[i,0], matrice[i,1], label=g, alpha=0.7, edgecolors='none')

        self.ax.legend()
        self.ax.grid(True)

        #self.ax.scatter(matrice[:,0], matrice[:,1])
        #plt.show()
        self.fig.savefig('server/plot.png')
        print(self.ax.dataLim)
        print(self.ax.viewLim)
        print(self.ax._position)
        print(self.ax.figbox)
        #import pdb;pdb.set_trace()
        #dataset=load_data(hp)

    def on_get(self, req, res):
        img=plt.imread('server/plot.png')
        out = io.BytesIO()
        plt.imsave(out, img)
        res.data = out.getvalue()
        res.content_type = 'image/png'


def map_range(x, x0, x1, y0, y1):
    '''
    Map the number n from the range x0,x1 to the range y0,y1
    '''
    print('x : {}, x0 : {}, x1 : {} '.format(x,x0,x1))
    #import pdb;pdb.set_trace()
    nRel=(x-x0)/(x1-x0)
    return nRel*(y1-y0)+y0

from interface import closest_node
class SynthesisResource:
    def __init__(self, hp, plot_data, codes, plotRes):
        self.hp=hp
        self.tts=tts_model(hp)
        self.plot_data=plot_data
        self.codes=codes
        self.plotRes=plotRes

    def on_get(self, req, res):
        print('Text to synthesize')
        if not req.params.get('text'):
            raise falcon.HTTPBadRequest()
        if not req.params.get('x'):
            raise falcon.HTTPBadRequest()
        if not req.params.get('y'):
            raise falcon.HTTPBadRequest()
        

        print(req.params.get('text'))
        print('x and y:')
        xRel=float(req.params.get('x'))
        yRel=float(req.params.get('y'))
        print(xRel)
        print(yRel)
        
        data_box=self.plotRes.ax.viewLim._points
        rel_box=self.plotRes.ax.figbox._points
        print('data box:')
        print(data_box)
        print('rel box:')
        print(rel_box)

        xData=map_range(xRel, rel_box[0,0], rel_box[1,0], data_box[0,0], data_box[1,0])
        yData=map_range(1-yRel, rel_box[0,1], rel_box[1,1], data_box[0,1], data_box[1,1])
        print('xData and yData:')
        print(xData)
        print(yData)

        #idx=0
        idx=closest_node(np.array([xData,yData]), self.plot_data)
        print(self.codes[idx,:])
        code=np.array([np.array([self.codes[idx,:]])])
        sentence=req.params.get('text')
        self.tts.synthesize(text=sentence, emo_code=code)

        print(self.tts.outdir)
        wav, samplerate = sf.read(self.tts.outdir+'/test.wav')

        out = io.BytesIO()
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        scipy.io.wavfile.write(out, self.hp.sr, wav.astype(np.int16))
        
        res.data = out.getvalue()
        res.content_type = 'audio/wav'

from synthesize import tts_model
from synthesize_with_latent_space import scatter_plot
from wsgiref import simple_server

class ICE_TTS_server:
    def __init__(self, hp, plot_data, codes, emo_cats=None, port=5000, parent=None):
        self.api = falcon.API()
        plotRes=PlotResource(hp, plot_data, codes, emo_cats)
        self.api.add_route('/synthesize', SynthesisResource(hp, plot_data, codes, plotRes))
        self.api.add_route('/plot', plotRes)
        self.api.add_route('/', UIResource())
        self.port=port
        print('Serving on port %d' % port)
        simple_server.make_server('0.0.0.0', port, self.api).serve_forever()



if __name__ == '__main__':
    from wsgiref import simple_server
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
  
    print('Serving on port %d' % args.port)
    simple_server.make_server('0.0.0.0', args.port, api).serve_forever()