# -*- coding: utf-8 -*-
""" A data oversampling widget for the Orange3.

    This is a data oversampling widget for Orange3, that implements SMOTE. 
    SMOTE stands for "Synthetic Minority Oversampling TEchnique". This is a 
    very useful technique for classifying data when the dataset is imbalanced.
    The package used is called "imbalanced-learn". 
    Source: "https://imbalanced-learn.readthedocs.io/en/stable/index.html"
    To run the addon, just install it using 'pip install -e .' from its package folder.
    Don't forget to first activate the orange environment.

    __author__ = Panagiotis Papadopoulos
    __date__ = July 2020
    __version__ = 0.1.0
    __type__ = Orange Addon
    __platform__ = Windows (Orange enviroment)
    __email__ = 'Panagiotis Papadopoulos' <panatronic@outlook.com>
    __status__ = Dev
"""

import sys
import numpy
from AnyQt.QtCore import Qt

from Orange.widgets import settings, widget, gui
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.data import Table

from imblearn.over_sampling import SMOTE

""" SMOTE Parameters
    class imblearn.over_sampling.SMOTE(
        sampling_strategy='auto',
        random_state=None,
        k_neighbors=5,
        n_jobs=None
    )
"""

SAMPLING_STRATEGY = [
    ("Auto", "auto"),
    ("All", "all"),
    ("Not majority", "not majority"),
    ("Minority", "minority"),
    ("Not minority", "not minority"),
]


class SMOTE_w(widget.OWWidget):
    name = "SMOTE"
    description = "generate oversamples to balance the distribution of classes within an dataset"
    icon = "icons/SMOTE.svg"
    priority = 10


    class Inputs:
        unbalancedDataset = Input("Unbalanced Dataset", Table)

    class Outputs:
        balancedDataset = Output("Balanced Dataset", Table)


    class_sampling = settings.Setting(0)
    random_seed = settings.Setting(0)
    nearest_neighbours = settings.Setting(1)    
    commitOnChange = settings.Setting(0)
    want_main_area = False


    def __init__(self):
        super().__init__()

        self.dataset = None
        self.balancedDataset = None

        # GUI
        infobox = gui.widgetBox(self.controlArea, "Info")
        self.infoa = gui.widgetLabel(infobox, "No data on input yet, waiting to get something.")
        
        self.infob = gui.widgetLabel(infobox, '')
        
        self.infoc = gui.widgetLabel(infobox, '')
        
        self.infod = gui.widgetLabel(infobox, '')

        statusbox = gui.widgetBox(self.controlArea, "Input status")
        self.infoe = gui.widgetLabel(statusbox, '')

        self.optionsBox = gui.widgetBox(self.controlArea, "SMOTE Options")
        gui.comboBox(
            self.optionsBox,
            self,
            "class_sampling",
            orientation=Qt.Horizontal,
            label="Sampling strategy: ",
            items=[d[0] for d in SAMPLING_STRATEGY],
            callback=[self.selection, self.checkCommit])
        gui.spin(
            self.optionsBox,
            self,
            "random_seed",
            minv=0,
            maxv=99,
            step=1,
            label="Random seed:",
            callback=[self.selection, self.checkCommit],
        )
        gui.spin(
            self.optionsBox,
            self,
            "nearest_neighbours",
            minv=1,
            maxv=100,
            step=1,
            label="k neighbours:",
            callback=[self.selection, self.checkCommit],
        )
        gui.checkBox(self.optionsBox, self, "commitOnChange",
                     "Commit data on selection change")
        gui.button(self.optionsBox, self, "Commit", callback=self.commit)
        self.optionsBox.setDisabled(True)

    @Inputs.unbalancedDataset
    def set_data(self, dataset):
        if dataset is not None:
            self.dataset = dataset
            self.optionsBox.setDisabled(False)

            self.numberOfInputInstances = len(self.dataset)
            self.infoa.setText("%d instances in input data set" % self.numberOfInputInstances)
            numOfclasses = len(self.dataset.domain.class_var.values)
            self.infob.setText("%d values in the categorical outcome" % numOfclasses)

            self.X_input = self.dataset.X
            self.y_input = self.dataset.Y

            self.minClassInstances = self.numberOfInputInstances
            instancesCounter = 0
            for c in range(numOfclasses):
                for i in self.y_input:    
                    if i == c:
                        instancesCounter += 1
                if instancesCounter <= self.minClassInstances:
                    self.minClassInstances = instancesCounter
                instancesCounter = 0
            self.minClassInstances-=1

            self.selection()
        else:
            self.dataset = None
            self.balancedDataset = None
            self.optionsBox.setDisabled(True)
            self.infoa.setText(
                "No data on input yet, waiting to get something.")
            self.infob.setText('')
            self.infoc.setText('')
            self.infod.setText('')
            self.infoe.setText('')
        
        self.commit()

    def selection(self):
        if self.dataset is None:
            return

        if self.nearest_neighbours > self.minClassInstances:
            self.nearest_neighbours = self.minClassInstances

        sm = SMOTE(
            sampling_strategy=SAMPLING_STRATEGY[self.class_sampling][1],
            random_state=self.random_seed,
            k_neighbors=self.nearest_neighbours,
            n_jobs=None)
        X_res, y_res = sm.fit_resample(self.X_input, self.y_input)
        
        numberOfOutputInstances = len(y_res)
        self.infoc.setText(
            "%d instances in output data set" % numberOfOutputInstances)
        oversampling_percentage = (
            (numberOfOutputInstances - self.numberOfInputInstances) /
            self.numberOfInputInstances) * 100
        self.infod.setText("{:.2f}% oversampling".format(oversampling_percentage))
        
        if oversampling_percentage == 0:
            self.infoe.setText("Attention! Input dataset is allready balanced.")
        else:
            self.infoe.setText('Input dataset is imbalanced.')

        self.balancedDataset = Table(self.dataset.domain, X_res, y_res)

    def commit(self):
        self.Outputs.balancedDataset.send(self.balancedDataset)
        return

    def checkCommit(self):
        if self.commitOnChange:
            self.commit()


if __name__ == "__main__":
    WidgetPreview(SMOTE_w).run(Table("iris-imbalanced"))
