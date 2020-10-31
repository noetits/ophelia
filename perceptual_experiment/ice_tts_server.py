import falcon
import argparse
import numpy as np
import soundfile as sf
# import scipy
# from scipy.io.wavfile import write
import io
import json
import os


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('pdf')
from datetime import datetime

from wsgiref import simple_server


class WebResource:
    def __init__(self, web_path):
        self.web_path=web_path
    def on_get(self, req, res):
        res.content_type = 'text/html'
        with open (self.web_path, "r") as myfile:
            html_body=myfile.read()
        res.body = html_body

class StaticResource(object):
    def on_get(self, req, resp, filename):
        # do some sanity check on the filename
        resp.status = falcon.HTTP_200
        resp.content_type = 'appropriate/content-type'
        with open('all_with_pca_limits/'+filename, 'rb') as f:
            resp.body = f.read()



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

def map_range(x, x0, x1, y0, y1):
    '''
    Map the number n from the range x0,x1 to the range y0,y1
    '''
    print('x : {}, x0 : {}, x1 : {} '.format(x,x0,x1))
    print('y0 : {}, y1 : {} '.format(y0,y1))
    #import pdb;pdb.set_trace()
    nRel=(x-x0)/(x1-x0)
    return nRel*(y1-y0)+y0

# from scipy.spatial import distance
def closest_node(node, nodes):
    # closest_index = distance.cdist([node], nodes).argmin()
    closest_index=((nodes-node)**2).sum(axis=1).argmin()
    return closest_index

# def closest_node(node, nodes):
#     closest_index = distance.cdist([node], nodes).argmin()
#     return closest_index

class AudioResource:
    def __init__(self, plot_data, codes, plotRes, audio_path='/home/noetits/noe/work/blizzard_unsupervised_letters/synth/t2m3942_ssrn14/', default_text='The birch canoe slid on the smooth planks.'):
        self.sr=22050
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

        self.audio_path=audio_path

    def on_get(self, req, res):
        print('-----------------------------------')
        print('Text to synthesize')
        if not req.params.get('text'):
            print('No text was given, synthesizing default text.')
            raise falcon.HTTPBadRequest()
        if not req.params.get('x'):
            raise falcon.HTTPBadRequest()
        if not req.params.get('y'):
            raise falcon.HTTPBadRequest()
        
        print(req.params.get('text'))
        text=float(req.params.get('text'))
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
        
        
        print("Click detected :", xData, yData)

        # print(self.tts.outdir)

        wav, samplerate = sf.read('all_with_pca_limits/sent_'+str(int(text))+'_code_'+str(idx)+'.wav')

        out = io.BytesIO()
        # wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        # scipy.io.wavfile.write(out, self.sr, wav.astype(np.int16))
        # write(out, self.sr, wav)
        sf.write(out, wav, self.sr, format='wav')
        
        res.data = out.getvalue()
        res.content_type = 'audio/wav'
        # print(res.data)



class PlotResource:
    def __init__(self, plot_data, codes, emo_cats=None, n_polar_axes=0, gradients=None):
        self.plot_name='plot'
        self.plot_data=plot_data
        self.codes=codes
        self.scatter_plot(plot_data, emo_cats, n_polar_axes)
        self.n_polar_axes=n_polar_axes
    def scatter_plot(self,matrice, emo_cats=None, n_polar_axes=0):
        self.fig, self.ax = plt.subplots()
        # Calculate the point density
        #z = gaussian_kde(matrice)(matrice)
        #z-=np.min(z)
        #z/=np.max(z)
        
        if emo_cats!=None:
            import pandas as pd
            df=pd.DataFrame()
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

        self.fig.savefig('plot'+ self.plot_name +'.png')
        print(self.ax.dataLim)
        print(self.ax.viewLim)
        print(self.ax._position)
        print(self.ax.figbox)
        #import pdb;pdb.set_trace()
        #dataset=load_data(hp)

    def on_get(self, req, res):
        img=plt.imread('plot'+self.plot_name+'.png')
        out = io.BytesIO()
        plt.imsave(out, img)
        res.data = out.getvalue()
        res.content_type = 'image/png'



import pdb
class ICE_TTS_server:
    def __init__(self, plot_data, codes, emo_cats=None, n_polar_axes=0, model_type='t2m', port=5000, parent=None, web_page='web_page.html'):
        self.api = falcon.API()
        plotRes=PlotResource(plot_data, codes, emo_cats, n_polar_axes)
        # clickImg=ImgResource()
        #federImg=ImgResource('server/logo_FEDER+wallonie.jpg')
        
        self.api.add_route('/plot', plotRes)
        
        # pdb.set_trace()
        # self.api.add_route('/synthesizeSet', SynthesizeSet(hp, plot_data, codes, plotRes, model_type=model_type))
        # self.api.add_route('/', WebResource("server/experience_restricted.html"))
        self.api.add_route('/web_page', WebResource(web_page))
        self.api.add_route('/audio', AudioResource(plot_data, codes, plotRes))
        # self.api.add_route('/', WebResource("server/web_page.html"))

        # for el in os.listdir('all_with_pca_limits')[:2]:
        #     print(el)
        #     self.api.add_route('/static/'+el, StaticResource())
        self.api.add_route('/static/{filename}', StaticResource())

        
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
