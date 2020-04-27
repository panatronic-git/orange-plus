"""Synthetic Minority Over-sampling Technique (SMOTE).
    class imblearn.over_sampling.SMOTE(
      sampling_strategy='auto',
      random_state=None,
      k_neighbors=5,
      n_jobs=None
    )
    Parameters
    ----------
    categorical_features : ndarray of shape (n_cat_features,) or (n_features,)
        Specified which features are categorical. Can either be:
        - array of indices specifying the categorical features;
        - mask array of shape (n_features, ) and ``bool`` dtype for which
          ``True`` indicates the categorical features.
    sampling_strategy : float, str, dict or callable, default='auto'
        Sampling information to resample the data set.
        - When ``float``, it corresponds to the desired ratio of the number of
          samples in the minority class over the number of samples in the
          majority class after resampling. Therefore, the ratio is expressed as
          :math:`\\alpha_{os} = N_{rm} / N_{M}` where :math:`N_{rm}` is the
          number of samples in the minority class after resampling and
          :math:`N_{M}` is the number of samples in the majority class.
            .. warning::
               ``float`` is only available for **binary** classification. An
               error is raised for multi-class classification.
        - When ``str``, specify the class targeted by the resampling. The
          number of samples in the different classes will be equalized.
          Possible choices are:
            ``'minority'``: resample only the minority class;
            ``'not minority'``: resample all classes but the minority class;
            ``'not majority'``: resample all classes but the majority class;
            ``'all'``: resample all classes;
            ``'auto'``: equivalent to ``'not majority'``.
        - When ``dict``, the keys correspond to the targeted classes. The
          values correspond to the desired number of samples for each targeted
          class.
        - When callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples for each class.
    random_state : int, RandomState instance, default=None
        Control the randomization of the algorithm.
        - If int, ``random_state`` is the seed used by the random number
          generator;
        - If ``RandomState`` instance, random_state is the random number
          generator;
        - If ``None``, the random number generator is the ``RandomState``
          instance used by ``np.random``.
    k_neighbors : int or object, default=5
        If ``int``, number of nearest neighbours to used to construct synthetic
        samples.  If object, an estimator that inherits from
        :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the k_neighbors.
    n_jobs : int, default=None
        Number of CPU cores used during the cross-validation loop.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
        for more details.