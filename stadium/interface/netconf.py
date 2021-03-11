from PySide2 import QtWidgets, QtGui, QtCore, QtSvg
from stadium.settings import GlobalConfig as config
import os
from stadium.utils import Dense, Conv2D, Model, Flatten
from stadium.core.defaults import CustomCnnPolicy, CustomMlpPolicy


def get_icon(name):
    icon_path = os.path.join(config.ICONS, name+'.svg')
    return icon_path

class NetConfigWidget(QtWidgets.QWidget):
    def __init__(self, parent, name, config=None):
        super(NetConfigWidget, self).__init__(parent=parent)
        self.par = parent
        self.layers = []
        self.config = config
        self.flat = False
        self.initialized = False
        self.combo = None
        self.container = QtWidgets.QWidget(self)
        self.lay = QtWidgets.QVBoxLayout(self.container)
        self.lay.addStretch()
        self.display = QtSvg.QSvgWidget(self)
        self.display.setFixedHeight(150)
        self.main_lay = QtWidgets.QVBoxLayout(self)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.container)
        self.main_lay.addWidget(self.display)
        self.main_lay.addWidget(self.scroll)
        self.blank()
    
    def build(self, config, nn_type='CNN'):
        
        self.combo = NewLayer(self)
        self.layers = []
        self.container = QtWidgets.QWidget(self)
        self.lay = QtWidgets.QVBoxLayout(self.container)
        self.lay.addStretch()
        self.scroll.setWidget(self.container)

        self.config = config
        if 'Cnn' in nn_type:
            self.flat = False
            self.combo.model().item(2).setEnabled(True)
            for i, n_filters in enumerate(config.filters):
                layer = Conv(
                    parent=self, 
                    filters=n_filters, 
                    kernel=config.kernel_size[i], 
                    stride=config.stride[i],
                    n=i)
                self.add_layer(layer, update=False)
        else:
            self.flat = True
            self.combo.model().item(2).setEnabled(False)

        for i, nodes in enumerate(config.layers):
            layer = FC(self, nodes, n=i)
            self.add_layer(layer, update=False)

        self.lay.addWidget(self.combo)
        self.lay.addStretch()
        self.update_image()
        self.initialized = True

    
    def add_layer(self, obj, update=True):

        index = len(self.layers)
        if type(obj) is Conv and update:
            index = [type(x) is FC for x in self.layers].index(True)
        obj.update(index)
        self.lay.insertWidget(index, obj)
        self.layers.insert(index, obj)
        # print([layer for layer in self.layers], index)
        if update:
            self.update_image()

    def delete_layer(self, layer):
        self.lay.removeWidget(layer)
        layer.deleteLater()
        self.layers.remove(layer)
        self.update_image()
        print([layer for layer in self.layers])

    def update_image(self):
        action_space = self.par.manager.env.get_attr('action_space')[0]
        if hasattr(action_space, 'n'):
            n_outputs = action_space.n
        else:
            n_outputs = action_space.shape[0]
        input_shape = self.par.manager.env.get_attr('observation_space')[0].shape
        print(input_shape)
        if len(input_shape) <= 1:
            input_shape = (input_shape[0], 1, 1)
        imgpath = os.path.join(config.UTILS, 'net.svg')
        print(imgpath)
        model = Model(input_shape=input_shape)
        if self.flat:
            model.add(Flatten())
        for i, layer in enumerate(self.layers):
            model.add(layer.to_draw())
            if type(layer) is Conv and type(self.layers[i+1]) is FC:
                model.add(Flatten())
        model.add(Dense(n_outputs))

        model.save_fig(imgpath)
        self.display.load(imgpath)

    def blank(self):
        imgpath = os.path.join(config.UTILS, 'blank.svg')
        self.display.load(imgpath)

    def create_conf(self):
        conf = {}
        if not self.flat:
            filters, kernels, strides = [], [], []
            names = ['filters', 'kernel_size', 'stride']
            for layer in self.layers:
                if type(layer) is Conv:
                    filters += [layer.filters.val]
                    kernels += [layer.kernel.val]
                    strides += [layer.stride.val]
            conf = {**dict(zip(names, [filters, kernels, strides]))}

        fc_layers = []
        for layer in self.layers:
            if type(layer) is FC:
                fc_layers.append(layer.nodes.val)

        conf = {**conf, **{'layers':fc_layers}}
        if self.flat:
            return CustomMlpPolicy(**conf)
        else:
            return CustomCnnPolicy(**conf)

class ClickButton(QtWidgets.QPushButton):
    def __init__(self, parent, name, triggers, status=None, text=False):
        super(ClickButton, self).__init__(parent=parent)
        self.par = parent
        self.name = name
        self.setStatusTip(status)

        for trigger in triggers:
            self.clicked.connect(trigger)

        icon = QtGui.QIcon(parent=self)
        path = os.path.join(config.ICONS, name+'.svg')
        icon.addPixmap(QtGui.QPixmap(path), size=QtCore.QSize(30, 30)) 
        self.setIcon(icon)
        if text:
            self.setStyleSheet("text-align:left;padding:4px;")
            self.setText('  '+name)
        
class DialSpin(QtWidgets.QWidget):
    def __init__(self, par, name, max, val=0):
        super(DialSpin, self).__init__()
        self.par = par
        self.label = QtWidgets.QLabel()
        self.label.setText(name)
        self.spin = QtWidgets.QSpinBox()
        font = self.label.font()
        font.setPointSize(font.pointSize()-2)
        self.spin.setFont(font)
        self.label.setFont(font)
        self.spin.setRange(0, max)
        self.spin.setValue(val)
        self.lay = QtWidgets.QHBoxLayout(self)
        self.lay.setContentsMargins(-1,0,-1,0)
        self.lay.setSpacing(0)
        self.lay.addWidget(self.label)
        self.lay.addWidget(self.spin)
        self.spin.valueChanged.connect(self.par.update_image)

    @property
    def val(self):
        return self.spin.value()


class Layer(QtWidgets.QWidget):
    def __init__(self, parent, n=0):
        super(Layer, self).__init__(parent)
        self.par = parent
        self.type = self.__class__.__name__
        self.n = n
        self.label = QtWidgets.QLabel(self.type + ':'+ str(self.n))
        self.del_button = ClickButton(self, 'Delete', [self.delete], status='Delete Layer')
        self.del_button.setFixedSize(30, 30)
        self.lay = QtWidgets.QHBoxLayout(self)
        self.lay.setContentsMargins(-1,0,-1,0)
        font = self.label.font()
        font.setPointSize(font.pointSize()-2)
        self.label.setFont(font)
        self.lay.setSpacing(0)
        self.lay.addWidget(self.label)

    def delete(self):
        self.par.delete_layer(self)

    def __repr__(self):
        return self.type + str(self.n)
    
    def update(self, n):
        self.n = n
        self.label.setText(self.type + ':'+ str(self.n))

class Conv(Layer):
    def __init__(self, parent, filters=64, kernel=3, stride=1, padding='valid', n=0):
        super(Conv, self).__init__(parent, n)
        self.filters = DialSpin(self.par, 'Filters:', 512, val=filters)
        self.kernel = DialSpin(self.par, 'Kernel:', 15, val=kernel)
        self.stride = DialSpin(self.par, 'Stride:', 3, val=stride)

        self.padding = padding
        self.lay.addWidget(self.filters)
    
        self.lay.addWidget(self.kernel)
        self.lay.addWidget(self.stride) 
        self.lay.addWidget(self.del_button)

    def to_draw(self):
        k, s = self.kernel.val, self.stride.val
        lay = Conv2D(filters=self.filters.val, kernel_size=(k, k), strides=(s, s), padding=self.padding)
        return lay


class FC(Layer):
    def __init__(self, parent, nodes=128, n=0):
        super(FC, self).__init__(parent, n)
        self.nodes = DialSpin(self.par, 'Nodes:', 2048, val=nodes)
        self.lay.addWidget(self.nodes)
        self.lay.addWidget(self.del_button)

    def to_draw(self):
        lay = Dense(units=self.nodes.val)
        return lay

class NewLayer(QtWidgets.QComboBox):
    def __init__(self, parent):
        super(NewLayer, self).__init__()

        self.par = parent
        self.options = ['New Layer', 'Fully Connected', 'Convolution']
        self.mapping = dict(zip(self.options, [None, FC, Conv]))
        self.activated.connect(self.new_layer)
        self.add_options()

    def new_layer(self):
        i = self.currentIndex()-1
        object = self.mapping[self.currentText()]
        self.par.add_layer(object(self.par))
        self.setCurrentIndex(0)

    def add_options(self):
        for option in self.options:
            icon = QtGui.QIcon(QtGui.QPixmap(get_icon('Default')))
            self.addItem(icon, option)
        self.model().item(0).setEnabled(False)
