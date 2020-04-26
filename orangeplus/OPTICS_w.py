# -*- coding: utf-8 -*-
""" A data clustering widget for the Orange3.

    This is a data clustering widget for Orange3, that implements OPTICS algorithm.
    OPTICS stands for "Ordering Points To Identify the Clustering Structure". 
    This is a very useful algorithm for clustering data when the dataset is unlabeled 
    with Non-flat geometry or it has uneven cluster sizes or variable cluster density.
    The package used is the "sklearn". Source: "https://scikit-learn.org/stable/index.html"
    To run the addon, just install it with 'pip install -e .' from its package folder.
    Don't forget to activate orange environment first.

    __author__ = Panagiotis Papadopoulos
    __date__ = Feb 2020
    __version__ = 0.1.0
    __type__ = Orange Addon
    __platform__ = Windows (Orange enviroment)
    __email__ = 'Panagiotis Papadopoulos' <panatronic@outlook.com>
    __status__ = Dev
"""

import numpy as np
from AnyQt.QtCore import Qt
from AnyQt.QtGui import QColor

from Orange.widgets import widget, gui
from Orange.widgets import settings
from Orange.widgets.widget import Msg
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.utils.slidergraph import SliderGraph
from Orange.data import Table, Domain, DiscreteVariable

from pyqtgraph import mkPen
from pyqtgraph.functions import intColor

from sklearn.cluster import OPTICS
from sklearn.neighbors import VALID_METRICS


""" OPTICS Parameters
    class sklearn.cluster.OPTICS(
        *   min_samples=5,  {default=5 or int > 1}, title: Min samples 
            max_eps=inf,    {default=np.inf}, not changed
        *   metric='minkowski', {default='minkowski' or [1]}, title: Metric
            p=2,    {default=2}, not changed
            cluster_method='xi',    {default='xi'}, not changed
            eps=None,   {default=None}, not changed
        *   xi=0.05,    {default=0.05 or float, between 0 and 1}, title: Minimum steepness
            predecessor_correction=True,    {default=True}, not changed
            min_cluster_size=None,    {default=None}, not changed
        *   algorithm='auto',   {default=auto or ball_tree, kd_tree, brute, auto}, title: Algorithm for nearest neighbors:
            leaf_size=30,   {default=30}, not changed
            n_jobs=None,   {default=None}, not changed
    )
    
    [1] Valid values for metric are:

    from scikit-learn: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]

    from scipy.spatial.distance: [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, 
    ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]

    See the documentation for scipy.spatial.distance for details on these metrics.
"""

OPTICS_METRICS = [
    ("cityblock", "cityblock"),
    ("cosine", "cosine"),
    ("euclidean", "euclidean"),
    ("l1", "l1"),
    ("l2", "l2"),
    ("manhattan", "manhattan"),
    ("braycurtis", "braycurtis"),
    ("canberra", "canberra"),
    ("chebyshev", "chebyshev"),
    ("correlation", "correlation"),
    ("hamming", "hamming"),
    ("minkowski", "minkowski"),
    ("sqeuclidean", "sqeuclidean"),
]

OPTICS_ALGORITHM = [
    ("Auto","auto"),
    ("Ball Tree","ball_tree"),
    ("kd Tree","kd_tree"),
    ("Brute","brute"),
]

class OPTICS_w(widget.OWWidget):
    name = "OPTICS"
    description = "dynamicaly clustering unlabeled data by density"
    icon = "icons/OPTICS.svg"
    priority = 20

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        annotated_data = Output("Data", Table)

    class Error(widget.OWWidget.Error):
        not_enough_instances = Msg("Not enough unique data instances. "
                                   "At least two are required.")

    minimum_samples = settings.Setting(5)
    metric_methode = settings.Setting(11)
    xi_value = settings.Setting(0.05)
    algorithm_base = settings.Setting(0)
    auto_commit = settings.Setting(False)
    cut_point = xi_value
    want_main_area = True
    

    def __init__(self):
        super().__init__()

        self.data = None
        self.dataset = None
        self.annotated_data = None

        # GUI
        infobox = gui.widgetBox(self.controlArea, "Info")
        self.infoa = gui.widgetLabel(infobox, "No data on input yet, waiting to get something.")
        self.infob = gui.widgetLabel(infobox, "")
        self.infoc = gui.widgetLabel(infobox, "")
        self.infod = gui.widgetLabel(infobox, "")

        self.optionsBox = gui.widgetBox(self.controlArea, "OPTICS Options")
        gui.spin(
            self.optionsBox,
            self,
            "minimum_samples",
            minv=1,
            maxv=100,
            step=1,
            label="Core point neighbors ",
            callback=self._min_samples_changed
        )
        gui.comboBox(
            self.optionsBox,
            self,
            "metric_methode",
            orientation=Qt.Horizontal,
            label="Distance metric: ",
            items=[d[0] for d in OPTICS_METRICS],
            callback=self._metric_changed
        )
        gui.doubleSpin(
            self.optionsBox,
            self,
            "xi_value",
            minv=(0.000),
            maxv=(0.999),
            step=(0.001),
            label="Minimum steepness: ",
            callback=self._xi_changed
        )
        gui.comboBox(
            self.optionsBox,
            self,
            "algorithm_base",
            orientation=Qt.Horizontal,
            label="neighborhood algorithm: ",
            items=[d[0] for d in OPTICS_ALGORITHM],
            callback=self._algorithm_changed
        )
        self.optionsBox.setDisabled(True)
        
        gui.auto_apply(self.controlArea, self, "auto_commit")
        gui.rubber(self.controlArea)

        self.controlArea.layout().addStretch()

        self.plot = SliderGraph(
            x_axis_label="Ordering of the points as processed by OPTICS",
            y_axis_label="Reachability distance (epsilon distance)",
            callback=self._on_changed
        )

        self.mainArea.layout().addWidget(self.plot)

    def check_data_size(self, data):
        if data is None:
            return False
        if len(data) < 2:
            self.Error.not_enough_instances()
            return False
        return True

    def normalizing(self,model):
        clusters = [c if c >= 0 else np.nan for c in model.labels_]
        k = len(set(clusters) - {np.nan})
        clusters = np.array(clusters).reshape(len(self.data), 1)

        clust_var = DiscreteVariable("Cluster", values=["C%d" % (x + 1) for x in range(k)])

        domain = self.data.domain
        attributes, classes = domain.attributes, domain.class_vars
        meta_attrs = domain.metas
        x, y, metas = self.data.X, self.data.Y, self.data.metas

        meta_attrs += (clust_var, )
        metas = np.hstack((metas, clusters))

        domain = Domain(attributes, classes, meta_attrs)
        new_table = Table(domain, x, y, metas, self.data.W)

        # self.Outputs.annotated_data.send(new_table)
        return new_table

    def commit(self):
        self.cluster()
        return

    def cluster(self):
        if not self.check_data_size(self.data):
            return

        model = OPTICS(min_samples=self.minimum_samples, 
                                   metric=OPTICS_METRICS[self.metric_methode][1],
                                   xi=self.xi_value,
                                   algorithm=OPTICS_ALGORITHM[self.algorithm_base][1],
                                   )
        model.fit(self.data.X)
        self._plot_graph(model)
        self.result_OPTICS = self.normalizing(model)
        self.send_data()

    def _plot_graph(self,model):
        reachability = model.reachability_[model.ordering_]
        space = np.arange(len(reachability))
        reachability[reachability == np.inf] = np.nanmax(reachability[reachability != np.inf])
        labels = model.labels_[model.ordering_]
        cluster_count = (len(np.unique(labels[labels[:]>=0])))
        self.infoc.setText("%d values in the cluster outcome" % cluster_count)
        noisy_counter = len(space[labels==-1])
        self.infod.setText("%d noisy samples in the leaf cluster" % noisy_counter)
        
        x_plot = space
        y_plot = reachability
        self.plot.clear_plot()

        colors = np.arange(150, (150+cluster_count))
        for klaster, color in zip(range(0, cluster_count), colors):
            Xk = space[labels == klaster]
            Rk = reachability[labels == klaster]
            self.plot.plot(Xk, Rk, pen=mkPen(intColor(color), width=2), antialias=True)
        self.plot.plot(x_plot[labels==-1], y_plot[labels==-1], pen=mkPen(QColor('black'), width=2), antialias=True)

    @Inputs.data
    def set_data(self, dataset):
        self.Error.clear()
        if not self.check_data_size(dataset):
            self.optionsBox.setDisabled(True)
            self.plot.clear_plot()
            self.infoa.setText(
                "No data on input yet, waiting to get something.")
            self.infob.setText('')
            self.infoc.setText('')
            self.infod.setText('')
            self.dataset = None
            self.annotated_data = None
            self.Outputs.annotated_data.send(None)
            return

        self.data = dataset
        self.optionsBox.setDisabled(False)
            
        self.numberOfInputInstances = len(self.data)
        self.infoa.setText("%d instances in input data set" % self.numberOfInputInstances)
        numOfclasses = len(self.data.domain.class_var.values)
        self.infob.setText("%d values in the categorical outcome" % numOfclasses)
        
        self.commit()

    def checkCommit(self):
        if self.commitOnChange:
            self.commit()

    def send_data(self):
        self.Outputs.annotated_data.send(self.result_OPTICS)

    def _min_samples_changed(self):
        if self.data is None:
            return
        self.commit()

    def _metric_changed(self):
        if self.data is None:
            return
        self.algorithm_base = 0
        self.commit()

    def _xi_changed(self):
        self.commit()

    def _algorithm_changed(self):
        if self.data is None:
            return

        if self.algorithm_base != 0:
            if OPTICS_METRICS[self.metric_methode][1] not in VALID_METRICS[OPTICS_ALGORITHM[self.algorithm_base][1]]:
                self.algorithm_base = 0

        self.commit()

    def _on_changed(self, value):
        self.cut_point = value


if __name__ == "__main__":
    WidgetPreview(OPTICS_w).run(Table("iris-imbalanced"))
