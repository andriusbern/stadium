from PySide2 import QtCore, QtWidgets, QtGui
import sys, random, os, time
import numpy as np

from rl.interface import MainWidget
from rl.settings import GlobalConfig as config

class Window(QtWidgets.QMainWindow):
    def __init__(self, app):
        super(Window, self).__init__()
        self.setWindowTitle('Reinforcement Learning Interface')
        self.setWindowIcon(QtGui.QIcon(os.path.join(config.ICONS, 'App.svg')))
        self.setWindowIconText('RL GUI')
        self.app = app
        
        exit_action = QtWidgets.QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.setIcon(QtGui.QIcon(QtGui.QPixmap(os.path.join(config.ICONS, 'Exit'))))
        exit_action.triggered.connect(self.close)

        menu = self.menuBar()
        file = menu.addMenu('&File')
        actions = [exit_action]
        file.addActions(actions)
        help_menu = menu.addMenu('&Help')

        parameters = ['THRESHOLD', 'TIMESTEPS', 'DETECTION_SAMPLE_SIZE',
                      'SPECTROGRAM_LOW', 'SPECTROGRAM_HIGH', 'FREQUENCY_BANDS',
                      'TIMESTEPS', 'FFT_LENGTH']
        self.main_widget = MainWidget(self)

        self.setCentralWidget(self.main_widget)
        self.show()

class App(QtWidgets.QApplication):
    def __init__(self, *args):
        super(App, self).__init__(*args)
        self.setStyle('QtCurve')
        self.setFont(QtGui.QFont('Monospace', 11))
        self.window = Window(self)

    def closeEvent( self ):
        self.exit(0)
    
    def main(self, args):
        self.window.show()
        self.window.raise_()
        sys.exit(self.exec_())

if __name__ == "__main__":

    args = sys.argv
    interface = App(args)
    interface.main(args)
    