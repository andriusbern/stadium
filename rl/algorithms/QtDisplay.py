from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np
import sys, time, ast
import settings
import imageio, datetime
"""
Notes:
    Move all env dependencies to another object (most likely the trainer class)
"""

class QtDisplay(QtWidgets.QWidget):
    """
    pyqtgraph display that combines the State display along with reward plots
    various UI elements
    """
    def __init__(self):
        self.app = QtGui.QApplication(sys.argv)
        self.app.setApplicationName('Gym GUI')
        self.app.setStyle('Oxygen')
        super(QtDisplay, self).__init__()

        self.setGeometry(0,0,800,700)
        self.timer = QtCore.QTimer()
        self.testTimer = QtCore.QTimer()
        self.env = None
        self.initial = True
        self.setup()
        self.testState = None # Used for testing the environment
        self.recording = False
    
    def setup(self):
        """
        Create the GUI and instantiate its elements, assign methods to them
        """
        self.layout = QtWidgets.QGridLayout()
        pg.setConfigOption('background', 0.5)
        pg.setConfigOption('foreground', 'k')

        # Environment image
        self.view = pg.PlotItem()
        self.imageView = pg.ImageView(view=self.view)
        self.imageView.getView().invertY(False)                  # Flip Y axis
        self.imageView.getView().showGrid(True, True)            # Enable grid

        # Environment selection
        self.envLabel = QtWidgets.QLabel()
        self.envLabel.setText('Environment:')
        self.envSelectionDropdown = QtWidgets.QComboBox()       
        self.envSelectionDropdown.addItems(settings.ENVIRONMENTS)
        self.envSelectionDropdown.setCurrentText('Select the environment.')
        self.envSelectionDropdown.currentTextChanged.connect(self.envSelectionChanged)
        
        # Buttons
        self.initializeEnvironmentButton = QtWidgets.QPushButton()
        self.initializeEnvironmentButton.setText('Create Environment')
        self.initializeEnvironmentButton.clicked.connect(self.initializeEnv)

        self.startButton = QtWidgets.QPushButton('Run', self)
        self.startButton.setText('Train')
        self.startButton.clicked.connect(self.startLoop)

        self.pauseButton = QtWidgets.QPushButton('Pause')
        self.pauseButton.setText('Pause')
        self.pauseButton.clicked.connect(self.pause)

        self.stopButton = QtWidgets.QPushButton('Test')
        self.stopButton.setText('Test')
        self.stopButton.clicked.connect(self.testLoop)

        self.resetButton = QtWidgets.QPushButton('Reset', self)
        self.resetButton.setText('Reset')
        self.resetButton.clicked.connect(self.resetEnv)

        self.recordButton = QtWidgets.QPushButton('Record', self)
        self.recordButton.setText('Record')
        self.recordButton.clicked.connect(self.record)

        self.buttons =  [self.initializeEnvironmentButton,
                        self.startButton,
                        self.pauseButton,
                        self.stopButton,
                        self.resetButton,
                        self.recordButton]

        # Parameter summary
        
        self.statusBox = QtWidgets.QLineEdit()
        self.statusBox.setText('Select and create the environment.')

        # Parameter tables
        self.globalParameterTable = QtWidgets.QTableWidget(0, 1)
        self.learningParameterTable = QtWidgets.QTableWidget(0, 1)
        self.networkParameterTable = QtWidgets.QTableWidget(0,1)
        self.tables = [self.learningParameterTable, self.globalParameterTable, self.networkParameterTable]

        # Tooltips
        self.tooltipBox = QtWidgets.QTextEdit()
        self.tooltipBox.setText('Parameter tooltips')

        # Reward plot
        self.rewardWidget = pg.PlotWidget()
        self.rewardWidget.getPlotItem().setTitle('Reward')

        # Loss plot
        self.lossWidget = pg.PlotWidget()
        self.lossWidget.getPlotItem().setTitle('Policy loss')

        # Mean action plot (show the average value of actions {0;1} through episodes )
        self.actionWidget = pg.PlotWidget()
        self.actionWidget.getPlotItem().setTitle('Output of actor network (during testing)')
        self.actionScatter = pg.ScatterPlotItem(pen=pg.mkPen(width=10, color='r'), symbol='o', size=5)
        self.actionWidget.addItem(self.actionScatter)

        self.buttonStatus('uninitialized')
        self.envSelectionChanged()
        self.layout.setColumnMinimumWidth(5, 350)

        self.layout.addWidget(self.envLabel, 1, 1, 1, 1)
        self.layout.addWidget(self.envSelectionDropdown, 1, 2, 1, 2)
        self.layout.addWidget(self.initializeEnvironmentButton, 2, 1, 1, 3)
        self.layout.addWidget(self.startButton, 3, 1, 1, 1)
        self.layout.addWidget(self.pauseButton, 3, 2, 1, 1)
        self.layout.addWidget(self.stopButton, 3, 3, 1, 1)
        self.layout.addWidget(self.resetButton, 4, 1, 1, 2)
        self.layout.addWidget(self.recordButton, 4, 3, 1, 1)
        self.layout.addWidget(self.statusBox, 5, 1, 1, 3)
        self.layout.addWidget(self.rewardWidget, 6, 1, 2, 3)
        self.layout.addWidget(self.lossWidget, 10, 1, 3, 3)
        self.layout.addWidget(self.actionWidget, 15, 1, 3, 3)
        self.layout.addWidget(self.networkParameterTable, 1, 5, 5, 1)
        self.layout.addWidget(self.learningParameterTable, 6, 5, 2, 1)
        self.layout.addWidget(self.globalParameterTable, 10, 5, 3, 1)
        self.layout.addWidget(self.tooltipBox, 15, 5, 3, 1)
        self.setLayout(self.layout)
        self.initial = False

    def buttonStatus(self, mode):
        """
        Sets the status of the buttons
            self.buttons = [self.initializeEnvironmentButton,
                            self.startButton,
                            self.pauseButton,
                            self.stopButton,
                            self.saveButton,
                            self.loadButton,
                            self.resetButton]
        """
        buttonStatus = {'uninitialized': [1, 0, 0, 0, 0, 0],
                        'initialized':   [0, 1, 0, 1, 1, 0],
                        'training'   :   [0, 0, 1, 0, 0, 0],
                        'testing'    :   [0, 0, 1, 0, 0, 1],
                        'recording'    : [0, 0, 0, 0, 0, 1],
                        'paused'     :   [0, 1, 0, 1, 1, 0]
        }
        tableStatus = {'uninitialized':  [True, True, True],
                        'initialized' :  [True, True, False],
                        'recording'   :  [False, False, False],
                        'paused'     :   [True, True, False],
                        'training'   :   [True, True, False],
                        'testing'    :   [True, True, False]
        }

        for button, state in zip(self.buttons, buttonStatus[mode]):
            button.setEnabled(bool(state))

        for table, state in zip(self.tables, tableStatus[mode]):
            table.setEnabled(state)
        
    
    def envSelectionChanged(self):
        envName = self.envSelectionDropdown.currentText()
        try:
            self.parameters = settings.parameters[envName.strip()]
        except:
            self.parameters = settings.parameters['Default']
        self.resetTables()

    def resetTables(self):
        """
        Resets the parameter tables based on the environment currently selected
        """
        try:
            for table in self.tables:
                table.itemChanged.disconnect(self.tableChanged)
        except:
            pass
        self.setData(self.learningParameterTable, 'Learning')
        self.setData(self.globalParameterTable,   'Rendering')
        self.setData(self.networkParameterTable,  'Networks')
        for table in self.tables:
            table.itemChanged.connect(self.tableChanged)

    def setData(self, table, parametersDictName):
        """
        Sets the table to a dictionary of values
        """
        parameters = self.parameters[parametersDictName]
        headers = []
        for n, key in enumerate(parameters.keys()):
            headers.append(key)
            val = parameters[key]
            val = str(val)
            item = QtWidgets.QTableWidgetItem(val)
            if self.initial:
                table.insertRow(table.rowCount())
            table.setRowHeight(n, 10)
            table.setItem(n, 0, item)
        table.itemPressed.connect(self.showParameterTooltip)
        table.setHorizontalHeaderLabels([parametersDictName])
        table.setColumnWidth(0, 250)
        table.setVerticalHeaderLabels(headers)
        table.setFont(QtGui.QFont("Courier", 10))

    def tableChanged(self, item):
        """
        Change global parameters whenever the table is modified
        """
        row = item.row()
        parent = item.tableWidget()
        parameter = parent.verticalHeader().model().headerData(row, QtCore.Qt.Vertical)
        header = parent.horizontalHeader().model().headerData(0, QtCore.Qt.Horizontal)
        dataType = type(self.parameters[header][parameter])
        if dataType is list:
            # ast enables the conversion of '[1,2,3]' string to an actual list
            self.parameters[header][parameter] = ast.literal_eval(item.text())
        else:
            self.parameters[header][parameter] = dataType(item.text())
        
    def showParameterTooltip(self, item):
        """
        Upon selecting an item in the table, show a corresponding tooltip for the parameter description
        """
        row = item.row()
        parent = item.tableWidget()
        parameter = parent.verticalHeader().model().headerData(row, QtCore.Qt.Vertical)
        text = settings.tooltips[parameter]
        self.tooltipBox.setText(str(text))

    def updateImage(self, image=None):
        """
        Update the image display
        """
        if image is None:
            if settings.DRAW_SUPPORT:
                self.imageView.setImage(self.env.S.image[:,:,:1])
            else:
                self.imageView.setImage(self.env.S.image[:,:,0])
        else: 
            self.imageView.setImage(image)

    def updateReward(self, reward):
        """
        Override in the RL trainer subclass
        Update the reward plot
        """
        self.rewardWidget.getPlotItem().plot(clear=True).setData(reward)

    def updateLoss(self, loss):
        """
        Updates the loss of the policy network
        """
        self.lossWidget.getPlotItem().plot(clear=True).setData(loss)

    def updateActions(self, action):
        """
        Updates the mean action value plot
        """
        x = list(range(len(action)))
        y = action
        self.actionScatter.setData(x, y)

    def pause(self):
        """
        Pauses the timer and stops the current episode
        """
        self.timer.stop()
        self.testTimer.stop()
        self.statusBox.setText('Paused.')
        self.done = True
        self.buttonStatus('paused')
        self.recording = False

        # self.actionWidget.getPlotItem().clear()

    def resetEnv(self):
        """
        Reset env
        """
        self.timer.stop()
        self.testTimer.stop()
        self.done = True
        try:
            self.env.close()
            self.rewardWidget.plotItem().plot(clear=True)
            self.lossWidget.plotItem().plot(clear=True)
        except:
            pass
        try: nao_rl.destroy_instances()
        except: pass
        self.buttonStatus('uninitialized')
        self.terminate()
        self.rewardWidget.getPlotItem().clear()
        self.lossWidget.getPlotItem().clear()

    def startLoop(self):
        """
        Start the training loop

        .train method should be redefined in the subclass, otherwise this will sample actions randomly
        """
        self.timer.timeout.connect(self.train)
        self.timer.start(1)
        self.statusBox.setText('Training...')
        self.buttonStatus('training')

    def stopLoop(self):
        """
        Stop the training and reinitialize the loop
        """
        self.timer.stop()
        self.testTimer.stop()
        self.initializeEnv()
    
    def summary(self):
        """
        Summary of the training procedure,
        To be overriden in the subclass
        """
        # return 'Env: {}, \n Sampling random actions...'.format(self.env.name)

    def testLoop(self):
        self.testTimer.timeout.connect(self.test)
        self.testTimer.start(2)
        self.statusBox.setText('Testing current policy...')
        self.buttonStatus('testing')

    def record(self):
        """
        Records the agent in OpenAI gym envs
        """
        try: 
            if 'EnvSpec' in str(self.env.spec):
                if not self.recording:
                    print('Recording...')
                    rec = True
                    self.buttonStatus('recording')
                if self.recording:
                    print('Recording stopped.')
                    rec = False
                    if len(self.images) != 0:
                        name = self.envSelectionDropdown.currentText().strip()
                        time = datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')
                        imageio.mimsave('{}_{}.gif'.format(name, time), self.images, duration=.005)
                        print('{}.gif saved at the main directory.'.format(name))
                        self.images = []
                        self.pause()

                self.recording = rec
        except:
            print('Cannot record this environment (OpenAI gym environments only).')


    def train(self):
        """
        Override in the RL trainer subclass
        This base method randomly samples actions
        """
        
        stime = time.time()
        if self.env.done:
            self.env.reset()
        
        action = self.env.action_space.sample()
        state, reward, done, comp = self.env.step(action)
        self.updateImage()
        if self.env.steps % 1000 == 0:
            print(self.env.steps)

    def keyPressEvent(self, event):
        """
        Close the app with Esc
        """
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
            self.pause()
            try:
                self.env.close()
            except:
                pass

    def main(self):
        """
        Launch the app and start the training loop
        """
        self.show()
        self.raise_()
        sys.exit(self.app.exec_())

if __name__ == '__main__':
    display = gymQtDisplay()
    display.initializeEnv(env)
    display.main()