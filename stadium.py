from PySide2 import QtWidgets, QtGui, QtCore
import sys, os, time
from multiprocessing import Pool

from stadium.interface import Interface
os.environ['QT_MAC_WANTS_LAYER'] = '1'

def long_init(args):
    time.sleep(args)
    return 0

class MySplashScreen(QtWidgets.QSplashScreen):
    def __init__(self, animation, flags):
        QtWidgets.QSplashScreen.__init__(self, QtGui.QPixmap(), flags)
        self.setPixmap(QtGui.QPixmap(os.path.join(config.UTILS, 'loading.jpg')))

class Window(QtWidgets.QMainWindow):
    def __init__(self, app):
        super(Window, self).__init__()
        self.setWindowTitle('Stadium: Reinforcement learning interface')
        self.setWindowIcon(QtGui.QIcon(os.path.join(config.ICONS, 'stadium.ico')))
        self.setWindowIconText('RL GUI')
        self.app = app
        
        self.status_bar = self.statusBar()
        self.main_widget = Interface(self)
        self.setCentralWidget(self.main_widget)
        self.show()

if __name__ == "__main__":
    from stadium.settings import GlobalConfig as config
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    args = sys.argv
    interface = QtWidgets.QApplication(args)
    # interface.setStyle('QtCurve')
    interface.setFont(QtGui.QFont('Helvetica', 9))
    # interface.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    interface.setStyle('Fusion')
    splash = MySplashScreen(os.path.join(config.UTILS, 'loading.gif'), QtCore.Qt.WindowStaysOnTopHint)
    splash.show()
    splash.raise_()
    interface.processEvents()

    initLoop = QtCore.QEventLoop()
    pool = Pool(processes=1)
    pool.apply_async(long_init, [0.5], callback=lambda exitCode: initLoop.exit(exitCode))
    initLoop.exec_()

    
    window = Window(interface)
    window.show()
    window.resize(600, 600)
    window.raise_()
    splash.finish(window)
    sys.exit(interface.exec_())
    