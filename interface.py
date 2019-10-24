import sys
import os
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QLineEdit
from PyQt5.QtMultimedia import QSound

import matplotlib
matplotlib.use("Qt5Agg")

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from synthesize import tts_model

import numpy as np
from scipy.spatial import distance
def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return closest_index


class ICE_TTS(QDialog):

    def __init__(self, hp, plot_data, codes, parent=None):
        super(ICE_TTS, self).__init__(parent)

        self.codes=codes

        self.tts=tts_model(hp)
        self.textbox = QLineEdit(self)
        # self.textbox.move(20, 20)
        self.textbox.resize(280, 40)

        self.plot_data=plot_data

        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        #self.toolbar = NavigationToolbar(self.canvas, self)

        # Just some button connected to `plot` method
        #self.button = QPushButton('Plot')
        #self.button.clicked.connect(self.plot)

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.textbox)
        #layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        #layout.addWidget(self.button)
        self.setLayout(layout)

        self.plot()

    def plot(self):
        print('coucou')
        # instead of ax.hold(False)
        self.figure.clear()

        # create an axis
        ax = self.figure.add_subplot(111)
        ax.scatter(self.plot_data[:,0], self.plot_data[:,1])

        def onclick(event):
                print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                      ('double' if event.dblclick else 'single', event.button,
                       event.x, event.y, event.xdata, event.ydata))
                x = event.xdata
                y = event.ydata
                idx=closest_node(np.array([x,y]), self.plot_data)
                print('idx:')
                print(idx)
                print(self.codes[idx,:])
                code=np.array([np.array([self.codes[idx,:]])])
                sentence = self.textbox.text()
                print(sentence)
                self.tts.synthesize(text=sentence, emo_code=code)

                sound=QSound(os.path.join(self.tts.outdir, 'test.wav'))
                #sound=QtGui.QSound("demo/test.wav")
                sound.play()
        
        cid = self.figure.canvas.mpl_connect('button_press_event', onclick)

        # refresh canvas
        self.canvas.draw()

def main_work():
    app = QApplication(sys.argv)
    i=ICE_TTS(1)
    i.show()
    sys.exit(app.exec_())

if __name__=="__main__":

    main_work()