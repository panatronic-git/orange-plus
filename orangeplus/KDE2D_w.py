# -*- coding: utf-8 -*-
""" A visualization widget for Orange3.

    This is a visualization widget for Orange3, that displays a joint distribution
    of two attributes from a dataset.  The Widget is a two-dimensional kernel-density 
    estimate graph using Gaussian kernels. The Kernel density estimation (KDE) is a 
    method to estimate the probability density function (PDF) of a random variable in
    a non-parametric way. This widget is useful in the case where there are similarities 
    in data, capable of being hidden in other charts such as scatter plot. In addition, 
    hidden clusters can be found, or it can even indicate whether these data form normal 
    distributions. The package used is the "SciPy". Source: "https://scipy.org/about.html".
    To run the addon, just install it with 'pip install -e .' from its package folder.
    Don't forget to activate orange environment first.

    __author__ = Panagiotis Papadopoulos
    __date__ = April 2020
    __version__ = 0.1.0
    __type__ = Orange Addon
    __platform__ = Windows (Orange enviroment)
    __email__ = 'Panagiotis Papadopoulos' <panatronic@outlook.com>
    __status__ = Dev
"""

import numpy as np
from AnyQt.QtWidgets import QListWidget

from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.data import Table, ContinuousVariable
from Orange.widgets import widget, gui, settings
from Orange.widgets.widget import Input

import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas


class KDE2D_w(widget.OWWidget):
    name = 'KDE-2D'
    description = "Visualization of two dimensional kernel-density estimate using Gaussian kernels" \

    icon = 'icons/KDE2D.svg'
    priority = 30

    class Inputs:
        data = Input("Data", Table)

    attrs = settings.Setting([])

    def __init__(self):
        self.all_attrs = []
        gui.listBox(self.controlArea, self, 'attrs',
                    labels='all_attrs',
                    box='Dataset attribute(s)',
                    selectionMode=QListWidget.ExtendedSelection,
                    callback=self.on_changed)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.mainArea.layout().addWidget(self.canvas)

    @Inputs.data
    def set_data(self, data):
        self.data = data = None if data is None else Table(data)
        self.all_attrs = []
        if data is None:
            self.plot.clear()
            return
        self.all_attrs = [(var.name, gui.attributeIconDict[var])
                          for var in data.domain.variables
                          if (var is not data and
                              isinstance(var, ContinuousVariable))]
        self.attrs = [0]
        self.on_changed()

    def on_changed(self):
        if not self.attrs or not self.all_attrs:
            return
        
        if len(self.attrs) != 2:
            return
        
        # discards the old graph
        self.figure.clear()

        attr_name = []
        for attr in self.attrs:
            attr_name.append(self.all_attrs[attr][0])

        x = np.ravel(self.data.X[:,[self.attrs[0]]])
        y = np.ravel(self.data.X[:,[self.attrs[1]]])

        deltaX = (max(x) - min(x))/3
        deltaY = (max(y) - min(y))/3

        xmin = min(x) - deltaX
        xmax = max(x) + deltaX

        ymin = min(y) - deltaY
        ymax = max(y) + deltaY

        # Create meshgrid
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = st.gaussian_kde(values)
        kde = np.reshape(kernel(positions).T, xx.shape)

        # create an axis
        ax = self.figure.gca()

        ax.imshow(np.rot90(kde), cmap='coolwarm', aspect='auto', extent=[xmin, xmax, ymin, ymax])
        cset = ax.contour(xx, yy, kde, colors='k')
        ax.clabel(cset, inline=1, fontsize=10)
        ax.set_xlabel(attr_name[0])
        ax.set_ylabel(attr_name[1])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title('Two Dimensional Gaussian Kernel Density Estimation')

        # refresh canvas
        self.canvas.draw()

if __name__ == "__main__":
    WidgetPreview(KDE2D_w).run(Table("iris"))
