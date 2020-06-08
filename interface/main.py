## Main windows here, 
from rl.settings import GlobalConfig as settings
import os
from PySide2 import QtWidgets, QtGui, QtCore, QtWebEngineWidgets
from .widgets import ToggleButton, ClickButton, GroupBox, Control, ImageDisplay, EnvironmentBrowser
from rl.baselines import Trainer
from rl.config import InstanceManager
import time
import pyqtgraph as pg
import numpy as np
try:
    import nao_rl
except:
    pass

class WorkerSignals(QtCore.QObject):
    finished = QtCore.Signal()
    error = QtCore.Signal(tuple)
    result = QtCore.Signal(object)
    progress = QtCore.Signal(int)

class Worker(QtCore.QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()  

    @QtCore.Slot()
    def run(self):
        try:
            self.fn(*self.args, **self.kwargs)
        except KeyboardInterrupt:
            pass
        finally:
            self.signals
    
    @QtCore.Slot()
    def stop(self):
        raise KeyboardInterrupt

class MainWidget(QtWidgets.QWidget):
    def __init__(self, parent):
        super(MainWidget, self).__init__(parent=parent)

        self.manager = InstanceManager()
        self.worker = None
        self.threadpool = QtCore.QThreadPool()
        self.rewards = [0]

        self.env_browser = EnvironmentBrowser(self, clicked_fn=self.select_environment)
        self.display = ImageDisplay(self)
        self.timer = QtCore.QTimer()
        self.web_timer = QtCore.QTimer()
        self.web_timer.setSingleShot(True)
        self.web_timer.timeout.connect(self.load_tensorboard)
        self.create_layout()

    def create_layout(self):

        self.console = QtWidgets.QTextEdit(self)
        self.console.setMinimumHeight(250)
        self.console_container = GroupBox('Console', self, [self.console])

        self.instance_browser = InstanceBrowser(self)
        self.instance_browser_container = GroupBox('Instance List', self, [self.instance_browser])
        
        self.tensorboard = QtWebEngineWidgets.QWebEngineView(self)
        self.code = QtWidgets.QTextEdit(self)
        self.code.setFont(QtGui.QFont('Monospace', 9))
        self.tab = QtWidgets.QTabWidget(self)
        self.tab.addTab(self.code, 'Config')
        self.tab.addTab(self.tensorboard, 'Tensorboard')
        self.tab.setMinimumHeight(400)

        self.bar_plot = BarPlot(self)
        self.reward_plot = pg.PlotWidget()
        self.reward_plot.setBackground(None)
        self.reward_plot.getPlotItem().showGrid(x=False, y=True, alpha=1)
        self.reward_plot.setMaximumHeight(200)
        self.plot_container = GroupBox('Action probabilities / Rewards', self, [self.bar_plot, self.reward_plot])

        self.buttons = Control(self)
        self.buttons.setMinimumWidth(650)
        
        lay = QtWidgets.QGridLayout(self)
        lay.addWidget(self.env_browser, 1, 1, 1, 1)
        lay.addWidget(self.instance_browser_container, 1, 2, 1, 1)
        lay.addWidget(self.buttons, 2, 1, 1, 2)
        lay.addWidget(self.display, 1, 3, 3, 2)
        lay.addWidget(self.tab, 4, 1, 3, 2)
        lay.addWidget(self.plot_container, 4, 3, 1, 2)
        lay.addWidget(self.console_container, 5, 3, 1, 2)
        lay.setRowMinimumHeight(1, 350)
        
    def select_environment(self, env):
        path = self.manager.set_env(env)
        self.read_config(path)
        self.instance_browser.set_path(self.manager.env_path)

    def select_instance(self, identifier):
        self.kill()
        path = os.path.join(self.manager.env_path, identifier)
        self.manager.load_instance(path)
        self.manager.tensorboard(browser=False)
        self.read_config(os.path.join(path, 'config.yml'))
        self.web_timer.start(5000)
        self.get_image()
        # self.bar_plot.create_plots(settings.get('n_workers', 'main'))
        # self.trainer.load_model()

    def create_new_instance(self):
        self.kill()
        self.manager.new_instance()
        self.manager.tensorboard(browser=False)
        self.web_timer.start(5000)
        self.get_image()
        # self.bar_plot.create_plots(settings.get('n_workers', 'main'))

    def kill(self):
        try: nao_rl.destroy_instances()
        except: pass
        try:
            self.manager.trainer.env.render(close=True)
        except:
            pass

    def delete_instance(self):
        pass

    def get_image(self):
        img = self.manager.trainer.prep()
        if img is not None:
            if img.shape[2] == 3 or img.shape[2] == 1:
                tiled = img
            else:
                tiled = self.tile_image(img)
            try:
                if not self.manager.trainer.is_custom:
                    self.manager.trainer.env.render('rgb_array', close=True)
            except:
                pass
            self.display.update(tiled)

    def tile_image(self, state):
        state = state.squeeze()
        n, h, w = state.shape
        grids = {
            1: [1, 1], 2: [1, 2], 3: [2, 2], 4: [2, 2],
            5: [2, 3], 6: [2, 3], 7: [3, 3], 8: [3, 3],
            9: [3, 3], 10:[3, 4], 11:[3, 4], 12:[3, 4],
            13:[4, 4], 14:[4, 4], 15:[4, 4], 16:[4, 4],
            17:[4, 5], 18:[4, 5], 19:[4, 5], 20:[4, 5],
            21:[5, 5], 22:[5, 5], 23:[5, 5], 24:[5, 5],
            25:[5, 5], 26:[5, 6], 27:[5, 6], 28:[5, 6],
            29:[5, 6], 30:[5, 6], 31:[6, 6], 32:[6, 6]}
        print(state.shape)
        x, y = grids[n]
        tiled = np.zeros([x*w, y*h])
        for i in range(x):
            for j in range(y):
                if i*x+j >= n:
                    break
                tiled[w*i:w*(i+1), h*j:h*(j+1)] = state[i*x+j, :, :]
                
        return tiled

    def train(self, status):
        if status:
            self.manager.trainer.stop = False
            self.worker = Worker(self.manager.trainer.train)
            self.manager.save_instance()
            self.threadpool.start(self.worker)
        else:
            self.manager.trainer.stop = True
            

    def test(self, status):
        if status:
            self.manager.trainer.prep()
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.loop)
            self.timer.start(1)
        else:
            self.rewards = [0]
            self.timer.stop()
            self.timer = QtCore.QTimer()
            # try:
                # self.manager.trainer.env.render(close=True) 
            # except:
                # pass

    def loop(self):
        state, prob, r, done = self.manager.trainer.step()
        if done[0]:
            self.rewards = [0]
        else:
            self.rewards.append(self.rewards[-1] + r[0])
        self.reward_plot.getPlotItem().plot(clear=True, pen=pg.mkPen('k', width=2)).setData(self.rewards, color='gray')
        
        if len(state.shape) > 2:
            if state.shape[2] == 3 or state.shape[2] == 1:
                tiled = state
            else:
                tiled = self.tile_image(state)
            if tiled.shape[1] < 1000:
                self.display.update(tiled)

    def reset(self):
        pass

    def load_tensorboard(self):
        self.tensorboard.load(QtCore.QUrl('http://localhost:6006/#scalars&_smoothingWeight=0.95'))
        self.tensorboard.setZoomFactor(0.75)

    def read_config(self, path):
        with open(path, 'r') as f:
            text = f.readlines()
        
        self.code.setPlainText(''.join(text))
    
    def save_config(self):
        with open(settings.CONF_PATH, 'w') as f:
            text = self.code.toPlainText()
            f.write(text)
        self.select_environment(self.manager.env_name)
        self.create_new_instance()


class InstanceBrowser(QtWidgets.QTreeView):
    def __init__(self, parent):
        super(InstanceBrowser, self).__init__(parent=parent)
        self.par = parent

        self.model = QtGui.QFileSystemModel()
        self.setModel(self.model)
        self.setAnimated(False)
        self.setIndentation(20)
        self.setSortingEnabled(True)
        self.setColumnHidden(1, True)
        self.setColumnHidden(2, True)
        self.setColumnWidth(0, 150)

    def set_path(self, directory):
        self.setRootIndex(self.model.index(directory))
        self.selectionModel().selectionChanged.connect(self.select_instance)
        self.model.setRootPath(directory)

    def select_instance(self, item):
        item = item.value(0).indexes()[0].data(0)
        self.par.select_instance(item)

class BarPlotContainer(QtWidgets.QWidget):
    def __init__(self, parent):
        super(BarPlotContainer, self).__init__(parent=parent)
        self.par = parent
        self.plots = []
        self.lay = QtWidgets.QGridLayout(self)

    def create_plots(self, count):
        for plot in self.plots:
            self.lay.removeWidget(plot)
            self.plots.remove(plot)
            plot.deleteLater()

        grids = {
            1: [1, 1], 2: [2, 1], 3: [2, 2], 4: [2, 2],
            5: [3, 2], 6: [3, 2], 7: [3, 3], 8: [3, 3],
            9: [3, 3], 10:[4, 3], 11:[4, 3], 12:[4, 3],
            13:[4, 4], 14:[4, 4], 15:[4, 4], 16:[4, 4]}
        x, y = grids[count]
        for i in range(x):
            for j in range(y):
                plot = BarPlot(self)
                self.lay.addWidget(plot, i, j, 1, 1)
                self.plots.append(plot)

    def update_plots(self, heights):
        for i, plot in enumerate(self.plots):
            plot.bar_plot(heights[i])


class BarPlot(pg.PlotWidget):
    """
    """
    def __init__(self, parent, width=150):
        pg.PlotWidget.__init__(self)
        self.bars = []
        self.getPlotItem().getAxis('bottom').setStyle(showValues=False)
        self.setBackground(None)
        self.setYRange(0, 1)
        self.getPlotItem().showGrid(True, True)
        self.getPlotItem().setLabel(axis='bottom', text='Action number')
        self.getPlotItem().setLabel(axis='left', text='Probability')

    def bar_plot(self, heights, width=.8, clear=True):
        if clear:
            for item in self.bars:
                self.removeItem(item)
                del item

        for i, height in enumerate(heights):
            bar = pg.BarGraphItem(x=[i+1], height=[height], width=width)
            self.bars.append(bar)
            self.addItem(bar)
