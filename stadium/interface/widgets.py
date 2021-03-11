## All required widgets          
from PySide2 import QtWidgets, QtGui, QtCore, QtWebEngineWidgets, QtSvg
import os
from stadium.settings import GlobalConfig as config
import pyqtgraph as pg
import numpy as np
import copy


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

def get_pixmap(icon_namestring, mode=None, function=None):
    path = os.path.join(config.ICONS, icon_namestring+'.svg')
    return QtGui.QPixmap(path, size=QtCore.QSize(30, 30))

def change_font_size(widget, increment):
    font = widget.font()
    font.setPointSize(font.pointSize()+increment)
    widget.setFont(font)
    return widget

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
    def __init__(self, parent, names, trigger, status=None, text=False, tip=None):
        super(ToggleButton, self).__init__(parent=parent)
        self.par = parent
        self.setCheckable(True)
        self.names = names
        self.status = status
        self.use_text = text
        self.status_change(False)
        self.clicked[bool].connect(getattr(self.par, trigger))
        self.clicked[bool].connect(self.status_change)
        if tip:
            self.setToolTip(tip)

        icon = QtGui.QIcon
        modes = [icon.Mode.Normal, icon.Mode.Normal, icon.Mode.Disabled]
        states = [icon.State.Off, icon.State.On, icon.State.Off]
        icon = icon(parent=self)
        for i, name in enumerate(self.names):
            icon.addPixmap(get_pixmap(name), modes[i], states[i])
        self.setIcon(icon)
        self.setStyleSheet(style)
        if self.use_text:
            self.setStyleSheet("text-align:center;padding:4px;font-size: 12px;")
            # self.setFixedSize(180,30)
            self.setMinimumHeight(25)
        else:
            self.setFixedSize(30, 30)
    
    def status_change(self, toggled):
        tip = self.names[1] if toggled else self.names[0]
        status = self.status[1] if toggled else self.status[0]
        self.setStatusTip(status)
        if self.use_text:
            self.setText('  '+tip)

class ClickButton(QtWidgets.QPushButton):
    def __init__(self, parent, name, triggers, status=None, text=False):
        super(ClickButton, self).__init__(parent=parent)
        self.par = parent
        self.name = name
        self.setStatusTip(status)
        if status:
            self.setToolTip(status)
        
        for trigger in triggers:
            trigger = getattr(self.par, trigger)
            self.clicked.connect(trigger)

        icon = QtGui.QIcon(parent=self)
        icon.addPixmap(get_pixmap(name)) #, fns[i]
        self.setIcon(icon)
        self.setStyleSheet(style)
        if text:
            self.setStyleSheet("text-align:center;padding:4px;font-size: 12px;")
            self.setText('  '+name)
            self.setMinimumHeight(25)
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
                    "font-weight: bold;" \
                    "font-size: 13px;}" \
                    "QGroupBox::title {" \
                    "background-color: transparent;" \
                    "color: rgb(0, 0, 0);" \
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

import cv2
class ImageDisplay(GroupBox):
    def __init__(self, parent, size=(200, 200), clean=True):
        self.display = pg.ImageView()
        self.display.getView().getViewWidget().setMinimumSize(*size)
        self.display.getImageItem()
        super(ImageDisplay, self).__init__('Environment Display', parent, [self.display])
        if clean:
            for element in ['menuBtn', 'roiBtn', 'histogram']:
                getattr(self.display.ui, element).hide()
        
    def update_image(self, image=None):
        if image is not None:
            self.display.setImage(image)

class EnvironmentBrowser(GroupBox):
    def __init__(self, parent, clicked_fn):
        self.clicked_fn = clicked_fn
        self.list = EnvTable(parent, clicked_fn)
        self.list.setMinimumHeight(300)
        self.save_config_button = ClickButton(
            parent=parent, 
            name='Choose Environment', 
            triggers=['create_environment'],
            status='Use this environment to train models',
            text=True)
        super(EnvironmentBrowser, self).__init__('Environment List', parent, [self.list, self.save_config_button], grid=[2,1])
        self.setFixedWidth(300)
        
    
class EnvTable(QtWidgets.QTableWidget):
    def __init__(self, parent, clicked_fn):
        QtWidgets.QTableWidget.__init__(self, 0, 2, parent=parent)
        self.par = parent
        self.img_size = 75
        self.setColumnWidth(0, self.img_size)
        self.setColumnWidth(1, 160)
        self.clicked_fn = clicked_fn

        self.cellClicked.connect(self.row_selected)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.reinit()
        self.populate()
        self.selectRow(1)
    
    def populate(self):
        for env in config.env_list:
            self.new_target(env)
        self.resizeColumnsToContents()
        self.horizontalHeader().setStretchLastSection(True)

    def new_target(self, target):
        row = self.rowCount()
        self.insertRow(row)
        self.setRowHeight(row, self.img_size)
        self.setColumnWidth(0, self.img_size)
        
        target = config.translated_envs[target]
        img = os.path.join(config.UTILS, target+'.png')
        if not os.path.isfile(img):
            self.setRowHeight(row, 30)
            
            img = os.path.join(config.UTILS,'black.jpg')
            img_size = 100
        else:
            img_size = self.img_size

        thumb = QtGui.QPixmap(img)
        thumb = thumb.scaled(self.img_size, img_size)
        label = QtWidgets.QLabel()
        label.setPixmap(thumb)
        
        widget = QtWidgets.QTableWidgetItem

        self.setCellWidget(row, 0, label)
        self.setItem(row, 1, widget('{}'.format(target)))

    def row_selected(self, item):
        if type(item) is not int:
            item = item.row()

        env = self.item(item, 1).text()
        self.clicked_fn(config.inv_translated_envs[env])

    def reinit(self):
        self.clear()
        self.clearContents()
        for row in reversed(range(self.rowCount())):
            self.removeRow(row)
        self.setHorizontalHeaderLabels(['Image', 'Environment'])

class InstanceBrowser2(QtWidgets.QWidget):
    def __init__(self, parent):
        super(InstanceBrowser2, self).__init__(parent=parent)
        self.par = parent
        
        self.models = []
        self.list = QtWidgets.QTableWidget(0, 4, parent=parent)
        self.list.setMinimumHeight(250)
        lay = QtWidgets.QGridLayout(self)
        
        self.create_instance_button = ClickButton(
            parent=parent, 
            name='Create New Agent',
            triggers=['create_instance'],
            status='Create a new agent using the current parameter set. Most parameters cannot be changed after initializing the agent.',
            text=True)
        self.reset_parameters_button = ClickButton(
            parent=parent, 
            name='Reset Parameters', 
            triggers=['new_instance'],
            status='Reset parameters to their default values in order to create a new agent.',
            text=True)

        self.list.horizontalHeader().setStyleSheet("QHeaderView {font: 8pt Arial;}")
        
        self.reinit()
        self.list.cellClicked.connect(self.row_selected)
        self.list.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.list.setHorizontalHeaderLabels(['Name', 'Trained #', 'Top reward', 'Created'])
        
        fnt = self.list.font()
        fnt.setPointSize(7)
        self.list.setFont(fnt)
        self.list.resizeColumnsToContents()
        self.list.horizontalHeader().setStretchLastSection(True)

        lay.addWidget(self.list, 1, 1, 1, 2)
        lay.addWidget(self.create_instance_button, 2, 1, 1, 1)
        lay.addWidget(self.reset_parameters_button, 2, 2, 1, 1)

    def populate(self):
        if len(self.models) > 0:
            self.reinit()
        self.models = self.par.manager.get_models()
        for env in self.models:
            self.new_target(env)
        self.list.resizeColumnsToContents()
        self.list.setColumnWidth(0, 60)
        self.list.horizontalHeader().setStretchLastSection(True)
        

    def new_target(self, target):
        if target not in self.models:
            self.models.append(target)
        row = self.list.rowCount()
        self.list.insertRow(row)

        tag = target.split('_')
        d, m, h = [tag[-2]] + tag[-1].split('-')
        date = '{},  {}:{}'.format(d, m, h)

        config_path = os.path.join(self.par.manager.env_path, target, 'config.yml')
        try:
            model_conf = self.par.manager.config.load_file(config_path)
            steps = str(model_conf['main']['steps_trained'])
        except:
            steps = '0'
    
        try:
            model_conf = self.par.manager.config.load_file(config_path)
            reward = str(round(float(model_conf['main']['highest_reward']), 2))
        except:
            reward = '0'
        

        name = self.par.manager.env_name
        table_widget = QtWidgets.QTableWidgetItem
        columns = [config.translated_envs[name], steps, reward, date]
        for i, text in enumerate(columns):
            widget = table_widget(text)
            font = widget.font()
            font.setPointSize(font.pointSize()-2)
            widget.setFont(font)
            self.list.setItem(row, i, widget)
        
    def reinit(self):
        self.list.clear()
        self.list.clearContents()
        for row in reversed(range(self.list.rowCount())):
            self.list.removeRow(row)
        self.list.setHorizontalHeaderLabels(['Name', 'Trained #', 'Top reward', 'Created'])

    def row_selected(self, item):
        if type(item) is not int:
            item = item.row()
        # self.list.selectRow(item)
        directory = self.models[item]
        self.par.select_instance(directory)

    def select_last(self):
        n = self.list.rowCount()
        self.list.selectRow(n-1)


class TensorboardView(QtWebEngineWidgets.QWebEngineView):
    def __init__(self):
        super(TensorboardView, self).__init__()
        self.timer = QtCore.QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._update)
    
    def delayed_load(self, delay_ms=2500):
        self.timer.start(delay_ms)

    def _update(self):
        self.load(QtCore.QUrl('http://localhost:6006/#scalars&_smoothingWeight=0.99'))
        self.setZoomFactor(0.6)

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
            plot.bar_plot(heights)

class BarPlot(pg.PlotWidget):
    """
    """
    def __init__(self, parent, width=150):
        pg.PlotWidget.__init__(self)
        self.bars = []
        self.getPlotItem().getAxis('bottom').setStyle(showValues=False)
        self.setBackground(None)
        self.setYRange(-1, 1)
        self.getPlotItem().showGrid(True, True)
        self.getPlotItem().setLabel(axis='bottom', text='Action number')
        self.getPlotItem().setLabel(axis='left', text='Probability/value')
        self.setMinimumHeight(200)


    def bar_plot(self, heights, width=.8, clear=True):
        if heights.ndim > 1:
            heights = heights.squeeze()
        if heights.ndim == 0:
            heights = np.array([heights])
        abs_h = np.abs(heights).sum()
        if abs_h < 0.33:
            heights = heights * 3
        elif abs_h > 3:
            heights = heights / abs_h.max()
        if clear:
            for item in self.bars:
                self.removeItem(item)
                del item

        for i, height in enumerate(heights):
            bar = pg.BarGraphItem(x=[i+1], height=[height], width=width)
            self.bars.append(bar)
            self.addItem(bar)

def parse_parameter(key, value):
    datatype = type(value)
    try:
        name = config.translated_parameters[key]
    except:
        name = key
    param_dict = dict(
        name=name,
        type=datatype.__name__)
    if datatype is list:
        merger = dict(value=value[0], values=value)
    elif datatype is float:
        merger = dict(value=value, step=value/10.)
    else:
        merger = dict(value=value)
    return {**param_dict, **merger}


from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
class ConfigContainer(QtWidgets.QWidget):
    def __init__(self, parent):
        super(ConfigContainer, self).__init__(parent=parent)
        self.par = parent
        self.initial_state = None
        self.config = None
        self.params = None
        self.param_dicts = []
        self.lay = QtWidgets.QGridLayout()
        self.tree = ParameterTree()
        self.lay.addWidget(self.tree)
        self.setLayout(self.lay)

    def build_from_config(self, instance_config):
        self.tree.clear()
        self.param_dicts = []
        self.config = instance_config
        self.initial_state = copy.deepcopy(instance_config)
        self.mapping = {
            'Main Parameters':self.config.main.__dict__, 
            'Environment Parameters':self.config.env,
            'Model Parameters':self.config.model.__dict__}

        for param_group, param_obj in self.mapping.items():
            params = dict(
                name=param_group,
                type='group',
                children=[])
            try:
                for param, value in param_obj.items():
                    param_dict = parse_parameter(param, value)
                    params['children'].append(param_dict)
            except:
                pass

            self.param_dicts.append(params)


        self.params = Parameter.create(
            name='params', 
            type='group', 
            children=self.param_dicts)
        self.params.sigTreeStateChanged.connect(self.change)
        self.tree.setParameters(self.params, showTop=False)

    
    def change(self, param, changes):
        print("tree changes:")
        
        for param, change, data in changes:
            path = self.params.childPath(param)
            if path is not None:
                parent, name = path
            to_change = self.mapping[parent]
            if type(to_change) is dict:
                try:
                    to_change[config.inv_translated_parameters[name]] = data
                except:
                    to_change[name] = data
            else:
                setattr(to_change, name, data)
            print('  Parent;    %s'% parent)
            print('  Parameter: %s'% name)
            print('  Change:    %s'% change)
            print('  Data:      %s'% str(data))
            print('  ----------')

            if parent == 'Environment Parameters':
                self.par.rebuild_env()
            
            if parent == 'Main Parameters':
                if name == 'policy':
                    self.par.policy_changed()
                if name == 'model':
                    self.par.model_changed(data)
            
        
    def hints(self, item):
        print(item)









# class ParameterSpinBox(QtWidgets.QWidget):
#     def __init__(self, parent, parameter, rnd=True):
#         super(ParameterSpinBox, self).__init__(parent=parent)

#         # self.obj = QtWidgets.QSpinBox if rnd else QtWidgets.QDoubleSpinBox
#         self.par = parent
#         self.parameter = parameter
#         self.translated = config.translate(parameter)
#         self.scale = config.ranges[parameter]
#         val = config.get(parameter)
        
#         self.spin_box = QtWidgets.QSpinBox()
#         self.spin_box.setAlignment(QtGui.Qt.AlignRight | QtGui.Qt.AlignVCenter)
#         self.slider = QtWidgets.QSlider(QtGui.Qt.Horizontal)
#         self.set_ranges()
#         self.spin_box.setValue(val)
#         self.spin_box.setMinimumWidth(80)
#         self.slider.setValue(self.find_nearest(val))
#         self.slider.valueChanged[int].connect(self.value_changed)
#         self.spin_box.valueChanged[int].connect(self.update_slider)
        
#         name = self.translated + ': '
#         self.label = QtWidgets.QLabel(name)
#         self.label.setAlignment(QtGui.Qt.AlignRight)
#         self.label.setMinimumWidth(130)
#         font = self.label.font()
#         font.setPointSize(font.pointSize()-2)
#         self.label.setFont(font)
#         self.spin_box.setFont(font)
#         layout = QtWidgets.QHBoxLayout(self)
#         layout.addWidget(self.label)
#         layout.addWidget(self.slider)
#         layout.addWidget(self.spin_box)
#         layout.setSpacing(10)
#         layout.setContentsMargins(10, 0, 10, 0)

#     def set_ranges(self):
#         self.spin_box.setRange(self.scale[0], self.scale[-1])
#         self.slider.setRange(0, len(self.scale)-1)
    
#     def update_slider(self):
#         value = self.find_nearest(self.spin_box.value())
#         self.slider.setValue(value)

#     def value_changed(self, value):
#         value = self.scale[self.slider.value()]
#         self.spin_box.setValue(value)
#         self.par.parameter_changed(self.parameter)
#         setattr(config, self.parameter, value)

#     def find_nearest(self, value):
#         array = np.asarray(self.scale)
#         idx = (np.abs(array - value)).argmin()
#         return idx

# class ParameterDial(QtWidgets.QWidget):
#     def __init__(self, parent, parameter, rnd=True):
#         super(ParameterDial, self).__init__(parent=parent)

#         # self.obj = QtWidgets.QSpinBox if rnd else QtWidgets.QDoubleSpinBox
#         self.par = parent
#         self.parameter = parameter
#         self.translated = config.translate(parameter)
#         self.scale = config.ranges[parameter]
#         val = config.get(parameter)
        
#         self.spin_box = QtWidgets.QSpinBox()
#         self.spin_box.setAlignment(QtGui.Qt.AlignCenter | QtGui.Qt.AlignVCenter)
#         self.slider = QSuperDial(self)
#         # sheet = "QDial{" \
#         #     "background-color: rgb(100, 100, 100);" \
#         #     "color: blue;" \
#         #     "}
#         self.slider.setStyleSheet(sheet)
#         self.set_ranges()
#         self.slider.setFixedSize(40, 40)
#         self.spin_box.setValue(val)
#         # self.spin_box.setMinimumWidth(80)
#         self.slider.setValue(self.find_nearest(val))
#         self.slider.valueChanged[int].connect(self.value_changed)
#         self.slider.setNotchesVisible(True)
#         # self.slider.setFixedSize(50, 50)
#         # self.slider.setAlignment(QtGui.Qt.AlignCenter)
#         self.spin_box.valueChanged[int].connect(self.update_slider)
#         self.slider.setWrapping(False)
#         name = self.translated + ': '
#         self.label = QtWidgets.QLabel(name)
#         self.label.setAlignment(QtGui.Qt.AlignCenter)
#         self.spin_box.setMaximumWidth(80)
#         # self.label.setMinimumWidth(130)
#         font = self.label.font()
#         font.setPointSize(font.pointSize()-3)
#         self.label.setFont(font)
#         self.spin_box.setFont(font)
#         layout = QtWidgets.QVBoxLayout(self)
#         layout.addWidget(self.label)
#         layout.addWidget(self.slider)
#         layout.addWidget(self.spin_box)
#         layout.setSpacing(5)
#         layout.setContentsMargins(5, 5, 5, 5)

#     def set_ranges(self):
#         self.spin_box.setRange(self.scale[0], self.scale[-1])
#         self.slider.setRange(0, len(self.scale)-1)
    
#     def update_slider(self):
#         value = self.find_nearest(self.spin_box.value())
#         self.slider.setValue(value)

#     def value_changed(self, value):
#         value = self.scale[self.slider.value()]
#         self.spin_box.setValue(value)
#         self.par.parameter_changed(self.parameter)
#         setattr(config, self.parameter, value)

#     def find_nearest(self, value):
#         array = np.asarray(self.scale)
#         idx = (np.abs(array - value)).argmin()
#         return idx


# class ParameterCheckBox(QtWidgets.QWidget):
#     def __init__(self, name, parent, fn):
#         super(ParameterCheckBox, self).__init__(parent=parent)
#         self.fn = fn
#         self.name = name
#         self.label = QtWidgets.QLabel(config.translate(name)+ ':')
#         self.label.setAlignment(QtGui.Qt.AlignRight)
#         self.check = QtWidgets.QCheckBox()
#         font = self.label.font()
#         font.setPointSize(font.pointSize()-2)
#         self.check.setChecked(config.get(name))
#         self.label.setFont(font)
#         self.check.clicked.connect(self.was_clicked)
#         self.main_layout = QtWidgets.QHBoxLayout(self)
#         self.main_layout.setSpacing(5)
#         self.main_layout.setContentsMargins(0, 0, 0, 0)
#         self.main_layout.addWidget(self.label)
#         self.main_layout.addWidget(self.check)
#         self.main_layout.setAlignment(QtGui.Qt.AlignRight)
#         self.setMinimumHeight(20)
    
#     def was_clicked(self):
#         setattr(config, self.name, not config.get(self.name))
#         self.fn(self.name)
