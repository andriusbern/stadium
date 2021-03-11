## Main windows here, 
from datetime import datetime
from stadium.settings import GlobalConfig as settings
import os
from PySide2 import QtWidgets, QtGui, QtCore
from .widgets import *
from stadium.core import InstanceManager
from stadium.environments import GymClassicWrapper
import pyqtgraph as pg
import numpy as np
from .netconf import NetConfigWidget
from stadium.interface import netconf

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

class WorkerSignals(QtCore.QObject):
    finished = QtCore.Signal()
    error = QtCore.Signal(tuple)
    result = QtCore.Signal(object)
    progress = QtCore.Signal(int)

class TrainingThread(QtCore.QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(TrainingThread, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()  
        self.kwargs['progress_callback'] = self.signals.progress

    @QtCore.Slot()
    def run(self):
        try:
            self.fn(*self.args, **self.kwargs)
        except KeyboardInterrupt:
            pass
        finally:
            self.signals.finished.emit()

    @QtCore.Slot()
    def stop(self):
        raise KeyboardInterrupt

class Interface(QtWidgets.QWidget):
    def __init__(self, parent):
        super(Interface, self).__init__(parent=parent)

        self.manager = InstanceManager()
        self.worker = None
        self.threadpool = QtCore.QThreadPool()
        self.rewards = [0]
        self.progress = 0
        self.display = ImageDisplay(self)
        self.timer = QtCore.QTimer()
        self.initialize()
        self.create_layout()
        self.highest_reward = -1000.

    def initialize(self):
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)

        self.train_button = ToggleButton(
            parent=self,
            names=['Train Agent', 'Stop Training'],
            trigger='train',
            status=['Start training the agent in the environment', 'Stop training'],
            text=True)

        self.test_button = ToggleButton(
            parent=self,
            names=['Test Agent', 'Stop Testing'],
            trigger='test',
            status=['Visualize the agent acting in the environment', 'Stop visualizing'],
            text=True)

        buttons = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(buttons)
        lay.addWidget(self.test_button)
        lay.addWidget(self.train_button)

        self.bar_plot = BarPlotContainer(self)
        self.bar_plot.create_plots(1)
        self.reward_plot = pg.PlotWidget()
        self.reward_plot.setBackground(None)
        self.reward_plot.getPlotItem().setLabel(axis='bottom', text='Step')
        self.reward_plot.getPlotItem().setLabel(axis='left', text='Reward')
        self.reward_plot.setXRange(0., 500)
        self.reward_plot.setYRange(-100, 100)
        self.reward_plot.getPlotItem().getViewBox().enableAutoRange(y=True, enable=True)
        
        self.reward_plot.setMinimumHeight(150)
        self.plot_container = GroupBox('Action probabilities / Rewards', self, [self.bar_plot, self.reward_plot], orientation='v')

        self.tensorboard_view = TensorboardView()
        self.plot_tab = QtWidgets.QTabWidget(self)
        self.plot_tab.addTab(self.plot_container, 'Rewards, actions')
        self.plot_tab.addTab(self.tensorboard_view, 'Tensorboard')

        self.console = QtWidgets.QTextEdit(self)
        self.text_log = ''
        self.control_container = GroupBox(
            'Agent control',
            self,
            [buttons, self.progress_bar], 
            orientation='v')

        self.instance_browser = InstanceBrowser2(self)
        self.env_browser = EnvironmentBrowser(self, clicked_fn=self.select_environment)
        self.netconf = NetConfigWidget(self, '')
        self.conf = ConfigContainer(self)
        self.tab = QtWidgets.QTabWidget(self)
        self.tab.addTab(self.netconf, 'Neural Networks')
        self.tab.addTab(self.conf, 'Hyperparameters')
        self.tab.setMinimumHeight(300)
        self.instance_browser_container = GroupBox('Agent configuration', self, [self.instance_browser, self.tab])
        self.env_browser.list.selectRow(0)
        self.manager.env_name = 'Pendulum-v0'
        self.log('Welcome to stadium! Please select your environment and configure an agent.')

    def create_layout(self):
        lay = self.lay = QtWidgets.QGridLayout(self)
        lay.addWidget(self.env_browser, 1, 1, 2, 1)
        lay.addWidget(self.instance_browser_container, 3, 1, 1, 2)
        lay.addWidget(self.display, 1, 2, 1, 1)
        lay.addWidget(self.control_container, 2, 2, 1, 1)
        lay.addWidget(self.plot_tab, 1, 3, 2, 1)
        lay.addWidget(self.console, 3, 3, 1, 1)
        lay.setRowMinimumHeight(1, 300)
        lay.setColumnMinimumWidth(2, 380)
        lay.setColumnMinimumWidth(3, 350)
        self.button_control(state=0)
        
    def select_environment(self, env):
        self.manager.env_name = env

    def create_environment(self):
        self.kill(all=True)
        path = self.manager.set_env(self.manager.env_name)
        self.conf.build_from_config(self.manager.config)
        self.netconf.blank()
        self.reset()
        self.netconf.build(self.manager.config.policy, nn_type=self.manager.config.main.policy)
        self.log('Initializing environment {}...'.format(self.manager.env_name))
        self.button_control(state=1)
        self.instance_browser.populate()
        self.manager.tensorboard(browser=False)
        self.tensorboard_view.delayed_load()
        self.ranges = [0, 0]
        self.manager.prep()
        state = self.manager.env.get_image()
        self.display.update_image(state)

    def select_instance(self, identifier):
        path = os.path.join(self.manager.env_path, identifier)
        self.manager.load_instance(path)
        self.load_instance_assets()
        self.reset()
        self.log('Selecting agent {}...'.format(identifier))
        self.button_control(state=2)

    def new_instance(self):
        self.log('Resetting parameters of {} environment...'.format(self.manager.config.env_name))
        path = self.manager.config.load(self.manager.config.env_name)
        self.load_instance_assets(image=False)
        self.button_control(1)
        self.instance_browser.list.clearSelection()

    def create_instance(self):
        self.manager.config.policy = self.netconf.create_conf()
        self.manager.config.main.steps_trained = 0
        self.manager.config.main.highest_reward = -1000.
        self.manager.new_instance()
        self.instance_browser.populate()
        self.load_instance_assets()
        self.reset()
        self.log('Saving new agent at directory: {}...'.format(self.manager.instance_path))
        self.button_control(state=2)
        self.instance_browser.select_last()

    def rebuild_env(self):
        self.manager.create_env(**self.manager.config.env)
        self.netconf.blank()
        self.netconf.build(self.manager.config.policy, nn_type=self.manager.config.main.policy)
        self.get_image()

    def load_instance_assets(self, image=True):
        self.conf.build_from_config(self.manager.config)
        self.netconf.blank()
        self.netconf.build(self.manager.config.policy, nn_type=self.manager.config.main.policy)
        if image:
            self.get_image()

    def model_changed(self, model):
        self.manager.config.load_model(model)
        self.conf.build_from_config(self.manager.config)

    def kill(self, all=False):
        try:
            if type(self.manager.env) is GymClassicWrapper or all:
                self.manager.env.close()
                del self.manager.model
        except:
            pass

    def delete_instance(self):
        pass

    def get_image(self):
        try:
            img = self.manager.prep()
            img = img.squeeze()
            self.display.update_image(img)
        except:
            pass

    def large_train(self, status):
        pass
        
    def train(self, status):
        if status:
            self.progress_bar.setRange(0, 100)
            self.progress = 0
            self.progress_bar.setValue(0)
            self.manager.stop_training = False
            self.worker = TrainingThread(self.manager.train)
            self.worker.signals.progress.connect(self.progress_callback)
            self.worker.signals.finished.connect(self.train_finished)
            self.manager.save_instance()
            self.threadpool.start(self.worker)
            self.log('Training agent for {} steps...'.format(self.manager.config.main.steps_to_train))
            self.plot_tab.setCurrentIndex(1)
            self.button_control(3)
        else:
            self.stop_training()
    
    def train_finished(self):
        self.log('Training finished.')
        self.button_control(2)
        self.stop_training()
        self.progress_callback(100)
        self.progress_bar.setValue(100)

    def stop_training(self):
        self.log('Stopping the training procedure, please wait...')
        self.manager.stop_training = True
        self.train_button.status_change(False)
        self.train_button.setChecked(False)

    def progress_callback(self, value=None):
        self.progress += value/self.manager.config.main.steps_to_train*100
        self.progress_bar.setValue(self.progress)
        item = self.instance_browser.list.item(self.instance_browser.list.currentRow(), 1)
        current = int(item.text())
        item.setText(str(current + int(value)))
            
    def test(self, status):
        if status:
            self.progress_bar.setRange(0, 200)
            self.plot_tab.setCurrentIndex(0)
            self.manager.prep()
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.loop)
            self.button_control(4)
            self.timer.start(1)
        else:
            self.manager.env.reset()
            self.rewards = [0]
            self.timer.stop()
            self.timer = QtCore.QTimer()
            self.button_control(2)
            self.progress = 0
            self.progress_bar.setValue(0)

    def loop(self):
        state, r, done, prob = self.manager.step()
        if len(self.rewards) > 200:
            done[0] = True
            self.manager.env.reset()
        if done[0]:
            self.check_rewards(self.rewards[-1])
            self.rewards = [0]
            self.progress = 0
        else:
            total_reward = self.rewards[-1] + r[0]
            self.rewards.append(total_reward)

        if int(self.progress) % 3 == 0:
            self.reward_plot.getPlotItem().plot(
                clear=True, pen=pg.mkPen('k', width=3)).setData(self.rewards, color='gray')
            self.bar_plot.update_plots(prob[0])

        self.display.update_image(state)
        self.progress += 1
        self.progress_bar.setValue(self.progress)

    def log(self, message):
        self.text_log += '[' + datetime.now().strftime("%H:%M:%S")+ ']: ' + message + '\n'
        self.console.setText(self.text_log)
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def button_control(self, state):
        buttons = [
            self.env_browser.save_config_button,
            self.instance_browser.reset_parameters_button,
            self.instance_browser.create_instance_button,
            self.test_button,
            self.train_button,
            self.netconf]

        states = {
            0: [0],
            1: [0, 1, 2, 5],
            2: [0, 1, 2, 3, 4],
            3: [4],
            4: [3]}

        indices = states[state]
        for button in buttons:
            button.setEnabled(False)
        for index in indices:
            buttons[index].setEnabled(True)

    def reset(self):
        try:
            self.highest_reward = self.manager.config.main.highest_reward
        except:
            self.highest_reward = -1000.
        self.progress = 0
        self.progress_bar.setValue(0)
        self.bar_plot.update_plots(np.array([0, 0]))

    def check_rewards(self, total_reward):
        print(total_reward)
        if total_reward > self.highest_reward:
            self.manager.config.main.highest_reward = float(total_reward)
            self.highest_reward = total_reward
            item = self.instance_browser.list.item(self.instance_browser.list.currentRow(), 2)
            item.setText(str(round(total_reward, 2)))