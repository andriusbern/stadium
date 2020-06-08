## All required widgets          
from PySide2 import QtWidgets, QtGui, QtCore
import os
from rl.settings import GlobalConfig as config
import pyqtgraph as pg
import numpy as np


def make_grid_layout(widgets, x, y):
    layout = QtWidgets.QGridLayout()
    for row in range(x):
        for col in range(y):
            widget = widgets[row*y+col]
            layout.addWidget(widget, row, col, 1, 1)

    return layout

def make_hv_layout(widgets, orientation):
    layouts = dict(h=QtWidgets.QHBoxLayout, v=QtWidgets.QVBoxLayout)
    layout = layouts[orientation]()
    for widget in widgets:
        layout.addWidget(widget)
    
    return layout


style = """ \
QPushButton{\
    background-color: grey;\
    border-style: outset;\
    border-width: 1px;\
    border-radius: 5px;\
    border-color: solid grey;\
    font: bold 12px;\
    padding: 6px;\
}\
QPushButton:pressed {\
    background-color: rgb(100, 100, 100);\
    border-style: inset;\
}\
QPushButton::menu-indicator {
    image: url(myindicator.png);
}
"""

class ToggleButton(QtWidgets.QPushButton):
    """
    Button with extended toggling functionality
    """
    def __init__(self, parent, names, trigger, status=None, text=False):
        super(ToggleButton, self).__init__(parent=parent)
        self.par = parent
        self.setCheckable(True)
        self.names = names
        self.status = status
        self.use_text = text
        self.status_change(False)
        self.clicked[bool].connect(getattr(self.par, trigger))

        icon = QtGui.QIcon
        modes = [icon.Mode.Normal, icon.Mode.Normal, icon.Mode.Disabled]
        fns = [icon.State.Off, icon.State.On, icon.State.Off]
        icon = icon(parent=self)
        for i, name in enumerate(self.names):
            path = os.path.join(config.ICONS, name+'.svg')
            print(path)
            icon.addPixmap(QtGui.QPixmap(path, size=QtCore.QSize(30, 30)), modes[i], fns[i])
        self.setIcon(icon)
        self.setStyleSheet(style)
        if self.use_text:
            self.setStyleSheet("text-align:left;\padding:4px;")
        else:
            self.setFixedSize(30, 30)

    def status_change(self, toggled):
        tip = self.status[1] if toggled else self.status[0]
        self.setStatusTip(tip)
        if self.use_text:
            self.setText('  '+tip)

    def stop(self):
        self.setChecked(False)

class ClickButton(QtWidgets.QPushButton):
    def __init__(self, parent, name, triggers, status=None, text=False):
        super(ClickButton, self).__init__(parent=parent)
        self.par = parent
        self.name = name
        self.setStatusTip(status)

        for trigger in triggers:
            trigger = getattr(self.par, trigger)
            self.clicked.connect(trigger)

        icon = QtGui.QIcon(parent=self)
        path = os.path.join(config.ICONS, name+'.svg')
        icon.addPixmap(QtGui.QPixmap(path), size=QtCore.QSize(30, 30)) #, fns[i]
        self.setIcon(icon)
        self.setStyleSheet(style)
        if text:
            self.setStyleSheet("text-align:left;padding:4px;")
            self.setText('  '+name)
        else:
            self.setFixedSize(30,30)
            # pass

class GroupBox(QtWidgets.QGroupBox):
    def __init__(self, name, parent, widgets=None, orientation='h', grid=[]):
        super(GroupBox, self).__init__(name, parent=parent)
        self.par = parent
        self.setFlat(True)
        self.setStyleSheet("QGroupBox {" \
                    "text-align: center;" \
                    "font-size: 12px;}" \
                    "QGroupBox::title {" \
                    "background-color: transparent;" \
                    "color: rgb(50, 50, 50);" \
                    "subcontrol-position: top center;"  
                    "} ")

        self.lay = None
        if widgets is not None:
            self.add_widgets(widgets, orientation, grid)

    def add_widgets(self, widgets, orientation, grid):
        if len(grid) > 1:
            x, y = grid
            self.setLayout(make_grid_layout(widgets, x, y))
        else:
            self.setLayout(make_hv_layout(widgets, orientation))


class Control(GroupBox):
    def __init__(self, parent):

        self.train_button = ToggleButton(
            parent=parent,
            names=['Train Model', 'Stop Training'],
            trigger='train',
            status=['Train model', 'Stop training'],
            text=True)

        self.test_button = ToggleButton(
            parent=parent,
            names=['Test', 'Stop'],
            trigger='test',
            status=['Test model', 'Stop testing'],
            text=True)

        self.create_instance_button = ClickButton(
            parent=parent, 
            name='New Instance', 
            triggers=['create_new_instance'],
            status='New Model Instance',
            text=True)

        self.save_config_button = ClickButton(
            parent=parent, 
            name='Save Config', 
            triggers=['save_config'],
            status='Save Config',
            text=True)
        
        self.delete_instance_button = ClickButton(
            parent=parent, 
            name='Delete Instance',
            triggers=['delete_instance'],
            status='Delete the current instance',
            text=True)

        self.reset_button = ClickButton(
            parent=parent, 
            name='Reset',
            triggers=['reset'],
            status='Reset to defaults',
            text=True)

        self.buttons = [
            self.train_button,
            self.test_button,
            self.create_instance_button,
            self.delete_instance_button,
            self.reset_button,
            self.save_config_button]
        
        super(Control, self).__init__(
            name='Control', 
            parent=parent,
            widgets=self.buttons,
            grid=[2, 3])
    

class ParameterSpinBox(QtWidgets.QWidget):
    def __init__(self, parent, parameter, rnd=True):
        super(ParameterSpinBox, self).__init__(parent=parent)

        # self.obj = QtWidgets.QSpinBox if rnd else QtWidgets.QDoubleSpinBox
        self.par = parent
        self.parameter = parameter
        self.translated = config.translate(parameter)
        self.scale = config.ranges[parameter]
        val = config.get(parameter)
        
        self.spin_box = QtWidgets.QSpinBox()
        self.spin_box.setAlignment(QtGui.Qt.AlignRight | QtGui.Qt.AlignVCenter)
        self.slider = QtWidgets.QSlider(QtGui.Qt.Horizontal)
        self.set_ranges()
        self.spin_box.setValue(val)
        self.spin_box.setMinimumWidth(80)
        self.slider.setValue(self.find_nearest(val))
        self.slider.valueChanged[int].connect(self.value_changed)
        self.spin_box.valueChanged[int].connect(self.update_slider)
        
        name = self.translated + ': '
        self.label = QtWidgets.QLabel(name)
        self.label.setAlignment(QtGui.Qt.AlignRight)
        self.label.setMinimumWidth(130)
        font = self.label.font()
        font.setPointSize(font.pointSize()-2)
        self.label.setFont(font)
        self.spin_box.setFont(font)
        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        layout.addWidget(self.spin_box)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 0, 10, 0)

    def set_ranges(self):
        self.spin_box.setRange(self.scale[0], self.scale[-1])
        self.slider.setRange(0, len(self.scale)-1)
    
    def update_slider(self):
        value = self.find_nearest(self.spin_box.value())
        self.slider.setValue(value)

    def value_changed(self, value):
        value = self.scale[self.slider.value()]
        self.spin_box.setValue(value)
        self.par.parameter_changed(self.parameter)
        setattr(config, self.parameter, value)

    def find_nearest(self, value):
        array = np.asarray(self.scale)
        idx = (np.abs(array - value)).argmin()
        return idx

class ParameterDial(QtWidgets.QWidget):
    def __init__(self, parent, parameter, rnd=True):
        super(ParameterDial, self).__init__(parent=parent)

        # self.obj = QtWidgets.QSpinBox if rnd else QtWidgets.QDoubleSpinBox
        self.par = parent
        self.parameter = parameter
        self.translated = config.translate(parameter)
        self.scale = config.ranges[parameter]
        val = config.get(parameter)
        
        self.spin_box = QtWidgets.QSpinBox()
        self.spin_box.setAlignment(QtGui.Qt.AlignCenter | QtGui.Qt.AlignVCenter)
        self.slider = QSuperDial(self)
        # sheet = "QDial{" \
        #     "background-color: rgb(100, 100, 100);" \
        #     "color: blue;" \
        #     "}
        self.slider.setStyleSheet(sheet)
        self.set_ranges()
        self.slider.setFixedSize(40, 40)
        self.spin_box.setValue(val)
        # self.spin_box.setMinimumWidth(80)
        self.slider.setValue(self.find_nearest(val))
        self.slider.valueChanged[int].connect(self.value_changed)
        self.slider.setNotchesVisible(True)
        # self.slider.setFixedSize(50, 50)
        # self.slider.setAlignment(QtGui.Qt.AlignCenter)
        self.spin_box.valueChanged[int].connect(self.update_slider)
        self.slider.setWrapping(False)
        name = self.translated + ': '
        self.label = QtWidgets.QLabel(name)
        self.label.setAlignment(QtGui.Qt.AlignCenter)
        self.spin_box.setMaximumWidth(80)
        # self.label.setMinimumWidth(130)
        font = self.label.font()
        font.setPointSize(font.pointSize()-3)
        self.label.setFont(font)
        self.spin_box.setFont(font)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        layout.addWidget(self.spin_box)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)

    def set_ranges(self):
        self.spin_box.setRange(self.scale[0], self.scale[-1])
        self.slider.setRange(0, len(self.scale)-1)
    
    def update_slider(self):
        value = self.find_nearest(self.spin_box.value())
        self.slider.setValue(value)

    def value_changed(self, value):
        value = self.scale[self.slider.value()]
        self.spin_box.setValue(value)
        self.par.parameter_changed(self.parameter)
        setattr(config, self.parameter, value)

    def find_nearest(self, value):
        array = np.asarray(self.scale)
        idx = (np.abs(array - value)).argmin()
        return idx



class ParameterCheckBox(QtWidgets.QWidget):
    def __init__(self, name, parent, fn):
        super(ParameterCheckBox, self).__init__(parent=parent)
        self.fn = fn
        self.name = name
        self.label = QtWidgets.QLabel(config.translate(name)+ ':')
        self.label.setAlignment(QtGui.Qt.AlignRight)
        self.check = QtWidgets.QCheckBox()
        font = self.label.font()
        font.setPointSize(font.pointSize()-2)
        self.check.setChecked(config.get(name))
        self.label.setFont(font)
        self.check.clicked.connect(self.was_clicked)
        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.main_layout.setSpacing(5)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.label)
        self.main_layout.addWidget(self.check)
        self.main_layout.setAlignment(QtGui.Qt.AlignRight)
        self.setMinimumHeight(20)
    
    def was_clicked(self):
        setattr(config, self.name, not config.get(self.name))
        self.fn(self.name)


class ImageDisplay(GroupBox):
    def __init__(self, parent, size=(400, 500), clean=True):
        self.display = pg.ImageView()
        # self.display.getView().getViewBox().setBackgound('w')
        self.display.getView().getViewWidget().setMinimumSize(*size)
        super(ImageDisplay, self).__init__('Environment Display', parent, [self.display])
        if clean:
            for element in ['menuBtn', 'roiBtn', 'histogram']:
                getattr(self.display.ui, element).hide()
        
    def update(self, image):
        image = np.flip(np.rot90(image/255., axes=(0, 1)), axis=1).astype(np.float32)
        self.display.setImage(image)

class EnvironmentBrowser(GroupBox):
    def __init__(self, parent, clicked_fn):
        self.clicked_fn = clicked_fn
        self.list = QtWidgets.QListWidget(parent)
        super(EnvironmentBrowser, self).__init__('Environment List', parent, [self.list])

        self.env_prefix = '   -'
        for env_type, env_list in config.envs.items():
        # for env_type, env_list in zip(env_types, env_lists):
            self.list.addItem(env_type + ' Environments')
            for env in env_list:
                self.list.addItem(self.env_prefix + env)

        self.list.itemClicked.connect(self.process_selection)

    def process_selection(self, item):
        string = item.text()
        if not string[:4] == self.env_prefix:
            return
        else:
            processed = string[len(self.env_prefix):]
            self.clicked_fn(processed)



        