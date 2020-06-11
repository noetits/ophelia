import falcon
import argparse
import numpy as np
import soundfile as sf
import scipy
from scipy.stats import gaussian_kde
import io
import json
import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('pdf')
from data_load import load_data
from datetime import datetime

from synthesize import tts_model
from wsgiref import simple_server



class SurveyResource(object):
    def on_post(self, req, resp):
        print(req)
        #import pdb;pdb.set_trace()
        table = req.bounded_stream.read().decode()
        dic_table=json.loads(table)

        # get the user and remove it
        user=dic_table[-1]
        dic_table.pop(-1)

        df_table=pd.DataFrame.from_records(dic_table)
        print(df_table)
        now=datetime.now().strftime("%m_%d_%Y_%H:%M:%S") 
        df_table.to_csv('server/surveys/'+user+'_'+now+'.csv')
        #value = req.get_param("json", required=True)
        #print(value)
        #do something with value

class WebResource:
    def __init__(self, web_path):
        self.web_path=web_path
    def on_get(self, req, res):
        res.content_type = 'text/html'
        with open (self.web_path, "r") as myfile:
            html_body=myfile.read()
        res.body = html_body

class ImgResource:
    def __init__(self,  path_to_img='server/CLICK_BU_POS_RVB.png'):
        self.path_to_img=path_to_img
        self.ext=path_to_img.split('.')[-1]
    def on_get(self, req, res):
        img=plt.imread(self.path_to_img)
        out = io.BytesIO()
        plt.imsave(out, img)
        res.data = out.getvalue()
        res.content_type = 'image/'+self.ext

class PlotResource:
    def __init__(self,  hp, plot_data, codes, emo_cats=None, n_polar_axes=0, gradients=None):
        self.hp=hp
        self.plot_data=plot_data
        self.codes=codes
        self.scatter_plot(plot_data, emo_cats, n_polar_axes)
        self.n_polar_axes=n_polar_axes
    def scatter_plot(self,matrice, emo_cats=None, n_polar_axes=0):
        self.fig, self.ax = plt.subplots()
        
        df=pd.DataFrame()

        # Calculate the point density
        #z = gaussian_kde(matrice)(matrice)
        #z-=np.min(z)
        #z/=np.max(z)
        
        if emo_cats!=None:
            df['style']=emo_cats
            for g in pd.unique(df['style']):
                s=g.split('_')[-1].lower()
                i = np.where(df['style'] == g)
                print(i)
                self.ax.scatter(matrice[i,0], matrice[i,1], label=s, alpha=0.5, edgecolors='none')
                self.ax.legend(bbox_to_anchor=(0.11, 0.65))
        else:
            self.ax.scatter(matrice[:,0], matrice[:,1], alpha=0.1, edgecolors='none')

        #if gradients!=None:
        #    print('here we plot gradients')
        
        self.ax.grid(True)

        if n_polar_axes:
            i = np.arange(0, 1, 1/n_polar_axes)
            theta = np.pi * i
            #rect = [0.1, 0.1, 0.8, 0.8]
            #self.ax_polar = self.fig.add_axes(rect, polar=True, frameon=False)

            # this corresponds do the extreme right of the plot.
            r=self.ax.viewLim._points[1,0]

            graduations=np.arange(11)/10*r*2-r

            for i,t in enumerate(theta):
                x=r*np.cos(t)
                y=r*np.sin(t)
                self.ax.plot([-x,x], [-y,y], color='r', linewidth=3)
                self.ax.annotate(str(i+1),(x*1.1,y*1.1), fontsize=14, fontweight='bold')

                for j,g in enumerate(graduations):
                    gx=g*np.cos(t)
                    gy=g*np.sin(t)
                    self.ax.scatter(gx, gy, color='tab:gray')
                    self.ax.annotate(str(j),(gx,gy), fontsize=11)
                

            #import pdb;pdb.set_trace()
        #self.ax.scatter(matrice[:,0], matrice[:,1])
        #plt.show()



        self.fig.savefig('server/plot'+self.hp.logdir.split('/')[-2] +'.png')
        print(self.ax.dataLim)
        print(self.ax.viewLim)
        print(self.ax._position)
        print(self.ax.figbox)
        #import pdb;pdb.set_trace()
        #dataset=load_data(hp)

    def on_get(self, req, res):
        img=plt.imread('server/plot'+self.hp.logdir.split('/')[-2] +'.png')
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

from scipy.spatial import distance
def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return closest_index

class SynthesisResource:
    def __init__(self, hp, plot_data, codes, plotRes, model_type='t2m', default_text='The birch canoe slid on the smooth planks.'):
        self.hp=hp
        self.tts=tts_model(hp, model_type=model_type)
        self.plot_data=plot_data
        self.codes=codes
        self.plotRes=plotRes
        self.default_text=default_text

        self.data_box=self.plotRes.ax.viewLim._points
        self.rel_box=self.plotRes.ax.figbox._points
        print('data box:')
        print(self.data_box)
        print('rel box:')
        print(self.rel_box)

    def on_get(self, req, res):
        print('Text to synthesize')
        if not req.params.get('text'):
            print('No text was given, synthesizing default text.')
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
        
        xData=map_range(xRel, self.rel_box[0,0], self.rel_box[1,0], self.data_box[0,0], self.data_box[1,0])
        yData=map_range(1-yRel, self.rel_box[0,1], self.rel_box[1,1], self.data_box[0,1], self.data_box[1,1])
        print('xData and yData:')
        print(xData)
        print(yData)

        idx=closest_node(np.array([xData,yData]), self.plot_data)
        print(self.codes[idx,:])
        code=np.array([self.codes[idx,:]])
        sentence=req.params.get('text')
        if sentence=="#":
            sentence=self.default_text
        try:
            self.tts.synthesize(text=sentence, emo_code=code)
        except ValueError:
            self.tts.synthesize(text=sentence, emo_code=np.array([code]))

        print(self.tts.outdir)
        wav, samplerate = sf.read(self.tts.outdir+'/test.wav')

        out = io.BytesIO()
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        scipy.io.wavfile.write(out, self.hp.sr, wav.astype(np.int16))
        
        res.data = out.getvalue()
        res.content_type = 'audio/wav'

class SynthesizeSet:
    def __init__(self, hp, plot_data, codes, plotRes, model_type='t2m', default_text='The birch canoe slid on the smooth planks.'):
        self.hp=hp
        self.tts=tts_model(hp, model_type=model_type)
        self.plot_data=plot_data
        self.codes=codes
        self.plotRes=plotRes
        self.default_text=default_text

        self.data_box=self.plotRes.ax.viewLim._points
        self.rel_box=self.plotRes.ax.figbox._points

        print('data box:')
        print(self.data_box)
        print('rel box:')
        print(self.rel_box)

    def on_get(self, req, res):
        if not req.params.get('axis'):
            raise falcon.HTTPBadRequest()
        if not req.params.get('index'):
            raise falcon.HTTPBadRequest()

        axis=int(req.params.get('axis'))
        index=int(req.params.get('index'))
        print('axis {}, index {}'.format(axis, index))
        # this corresponds do the extreme right of the plot
        r=self.plotRes.ax.viewLim._points[1,0]
        i = np.arange(0, 1, 1/self.plotRes.n_polar_axes)
        theta = np.pi * i

        # express xData and yData from the axes
        value_along_axis=map_range(index, 0, 10, -r, r)

        print("value_along_axis")
        print(value_along_axis)
        xData=value_along_axis*np.cos(theta[axis-1])
        yData=value_along_axis*np.sin(theta[axis-1])

        print('xData and yData:')
        print(xData)
        print(yData)
        
        #idx=0
        idx=closest_node(np.array([xData,yData]), self.plot_data)
        print('idx')
        print(idx)
        print(self.codes[idx,:])
        code=np.array([np.array([self.codes[idx,:]])])
        
        sentence=self.default_text
        print("sentence:")
        print(sentence)


        #id=sentence+str(code)
        id='_'.join(sentence.split(' '))+str(idx)

        # Here I want to synthesize only if it was not synthesized and saved beofre.
        # But the attribute outdir is created only after going through the synthesis function. So I have to try to see if it exists
        # if it does, I check if the wav file exists, and if not synthesize
        try:
            if not os.path.exists(os.path.join(self.tts.outdir,id+'.wav')):
                self.tts.synthesize(text=sentence, emo_code=code, id=id)
        except:
            self.tts.synthesize(text=sentence, emo_code=code, id=id)

        print(self.tts.outdir)
        wav, samplerate = sf.read(os.path.join(self.tts.outdir,id+'.wav'))

        out = io.BytesIO()
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        scipy.io.wavfile.write(out, self.hp.sr, wav.astype(np.int16))
        
        res.data = out.getvalue()
        res.content_type = 'audio/wav'


class ICE_TTS_server:
    def __init__(self, hp, plot_data, codes, emo_cats=None, n_polar_axes=0, model_type='t2m', port=5000, parent=None):
        self.api = falcon.API()
        plotRes=PlotResource(hp, plot_data, codes, emo_cats, n_polar_axes)
        clickImg=ImgResource()
        #federImg=ImgResource('server/logo_FEDER+wallonie.jpg')
        #SynthesizeSet(hp, plot_data, codes, plotRes, model_type=model_type)
        #import pdb;pdb.set_trace()
        self.api.add_route('/plot', plotRes)
        
        #self.api.add_route('/synthesizeSet', SynthesizeSet(hp, plot_data, codes, plotRes, model_type=model_type))
        #self.api.add_route('/', WebResource("server/experience_restricted.html"))
        self.api.add_route('/synthesize', SynthesisResource(hp, plot_data, codes, plotRes, model_type=model_type))
        self.api.add_route('/', WebResource("server/web_page.html"))

        self.api.add_route('/table_script.js', WebResource("server/table_script.js"))
        self.api.add_route('/survey', SurveyResource())
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
