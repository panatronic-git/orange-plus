# -*- coding: utf-8 -*-
""" A visualization widget for Orange3.

    This is a visualization widget for Orange3, that displays a joint distribution of two
    variables from a dataset.  The Widget is a two-dimensional kernel-density estimate graph
    using Gaussian kernels. The Kernel density estimation (KDE) is a method to estimate the
    probability density function (PDF) of a random variable in a non-parametric way.
    This widget is useful in cases where there are similarities in data, that are difficult to
    spot in other charts such as scatter plots. In addition, hidden clusters can be found, as
    well as indicate whether the data form normal distributions.
    The package used is called "SciPy". Source: "https://scipy.org/about.html".
    To run the addon, just install it using 'pip install -e .' from its package folder.
    Don't forget to first activate the orange environment.

    __author__ = Panagiotis Papadopoulos
    __date__ = April 2020
    __version__ = 0.1.0
    __type__ = Orange Addon
    __platform__ = Windows (Orange enviroment)
    __email__ = 'Panagiotis Papadopoulos' <panatronic@outlook.com>
    __status__ = Dev
"""

import numpy as np
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QListWidget

from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.data import Table, ContinuousVariable
from Orange.widgets import widget, gui, settings
from Orange.widgets.widget import Input

import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

""" gaussian_kde Parameters
    class scipy.stats.gaussian_kde(dataset, bw_method=None)
    
    dataset : array_like
    Datapoints to estimate from. In case of univariate data this is a 1-D array, 
    otherwise a 2-D array with shape (# of dims, # of data).

    bw_method : str, scalar or callable, optional

    The method used to calculate the estimator bandwidth. 
    This can be ‘scott’, ‘silverman’, a scalar constant or a callable. 
    If a scalar, this will be used directly as kde.factor. 
    If a callable, it should take a gaussian_kde instance as only parameter 
    and return a scalar. If None (default), ‘scott’ is used. 
    See Notes for more details.
"""

BW_METHOD = [
    ("Scott", "scott"),
    ("Silverman", "silverman"),
]

class KDE2D_w(widget.OWWidget):
    name = 'KDE-2D'
    description = "Visualization of two dimensional kernel-density estimate using Gaussian kernels" \

    icon = 'icons/KDE2D.svg'
    priority = 30

    class Inputs:
        data = Input("Data", Table)

    attrs = settings.Setting([])
    bw_methode = settings.Setting(0)

    def __init__(self):
        self.data = None
        self.all_attrs = []
        gui.listBox(self.controlArea, self, 'attrs',
                    labels='all_attrs',
                    box='Dataset attribute(s)',
                    selectionMode=QListWidget.ExtendedSelection,
                    callback=self.on_changed)
        self.optionsBox = gui.widgetBox(self.controlArea, "KDE-2D Options")
        gui.comboBox(
            self.optionsBox,
            self,
            "bw_methode",
            orientation=Qt.Horizontal,
            label="Bandwidth Method: ",
            items=[d[0] for d in BW_METHOD],
            callback=self._bw_methode
        )
        self.optionsBox.setDisabled(True)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.mainArea.layout().addWidget(self.canvas)

    @Inputs.data
    def set_data(self, data):
        self.data = data = None if data is None else Table(data)
        self.all_attrs = []
        if data is None:
            # discards the old graph
            self.figure.clear()
            self.optionsBox.setDisabled(True)
            return
        self.all_attrs = [(var.name, gui.attributeIconDict[var])
                          for var in data.domain.variables
                          if (var is not data and
                              isinstance(var, ContinuousVariable))]
        self.attrs = [0]
        self.optionsBox.setDisabled(False)
        self.on_changed()

    def _bw_methode(self):
        if self.data is None:
            return
        self.on_changed()

    def on_changed(self):
        if not self.attrs or not self.all_attrs:
            return
        if self.data is None:
            return
        
        if len(self.attrs) != 2:
            return
        
        # discards the old graph
        self.figure.clear()

        # Get names of attrs
        attr_name = []
        for attr in self.attrs:
            attr_name.append(self.all_attrs[attr][0])

        # Get data
        x = np.ravel(self.data.X[:,[self.attrs[0]]])
        y = np.ravel(self.data.X[:,[self.attrs[1]]])

        # Calc boundaries
        dX = (max(x) - min(x))/3
        xmin = min(x) - dX
        xmax = max(x) + dX

        dY = (max(y) - min(y))/3
        ymin = min(y) - dY
        ymax = max(y) + dY

        # Create meshgrid
        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

        # calc KDE
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([x, y])
        kernel = st.gaussian_kde(values, bw_method=BW_METHOD[self.bw_methode][1])
        
        # Calc Z
        Z = np.reshape(kernel(positions).T, X.shape)

        # create current axes
        ax = self.figure.gca()
        
        # create backaground
        ax.imshow(np.rot90(Z), cmap='coolwarm', aspect='auto', extent=[xmin, xmax, ymin, ymax])
        
        # create contour
        cset = ax.contour(X, Y, Z, colors='k')
        
        # create labels
        ax.clabel(cset, inline=1, fontsize=10)
        ax.set_xlabel(attr_name[0])
        ax.set_ylabel(attr_name[1])
        
        # set limits
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        
        # set Title
        ax.set_title('Two Dimensional Gaussian Kernel Density Estimation')

        # refresh canvas
        self.canvas.draw()

if __name__ == "__main__":
    WidgetPreview(KDE2D_w).run(Table("iris"))
