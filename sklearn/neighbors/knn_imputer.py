"""Nearest Neighbor Imputation"""

import numpy as np

from .base import _get_weights, _check_weights, NeighborsBase, KNeighborsMixin
from .base import RadiusNeighborsMixin, SupervisedFloatMixin
from ..base import RegressorMixin
from ..utils import check_array

class KNeighborsImputer(NeighborsBase, KNeighborsMixin,
                          SupervisedFloatMixin,
                          RegressorMixin):
    """ Imputation based on k-nearest neighbors.
    
    The missing values is predicted by local interpolation using the target 
    values of the nearest neighbors in the training data.
    
    Paramters
    ----------
    n_neighbors: int, optional (default = 5)
        Number of neighbors to use by default for :meth: `k_neighbors` queries.
        
    weights: str or callable (default = 'uniform')
        weight function used in prediction.Possible values:
        - 'uniform': uniform weights. All distances in the neighborhood will be 
           weighed equally.
        - 'distance': the weights of target values of the neighborhood points 
           will be the inverse of the distance from the point to be imputed. 
           Closer thee points, lesser will be their inter-point distance and will
           thus have larger influence. 
        - [callable]: A user definec funciton that takes a distance array as input
           and returns an array of the same shape containing the weights.
           
    algorithm: {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional (default = 'auto')
        Algorithm used to compute the nearest neighbors. 
        
        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDtree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.
        
    leaf_size: int, optinal (default = 30)
        Leaf size works with the tree based methods of finding the nearest 
        neighbors - KDTree and BallTree.
        
    metric: string or DistanceMetric object (default='minkowski')
        the metric to be used by the tree to calculate the distance between two 
        points. The default is 'minkowski', and with p=2 is equivalent to the 
        standard Euclidean metric. See the documentation of the DistanceMetric 
        class for a list of available metrics. 
        
    p: integer, optional (default = 2):
        Power parameter for the minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
        
    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.
    
    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run for neighbors search.
        If ``-1``, then the number of jobs is set to the number of CPU cores.
        Doesn't affect :meth:`fit` method.
    
    max_iter: int, optional (default = 10)
        the number of iterations across which the imputed values will be fine 
        tuned.
    """
    def __init__(self, n_neighbors=5, weights='uniform', 
                 algorithm='auto', leaf_size=30,
                p=2, metric='minkowski', metric_params=None, n_jobs = 1,
                max_iter=10):
        self._init_params(n_neighbors=n_neighbors,
                          algorithm=algorithm,
                          leaf_size=leaf_size, metric=metric, p=p,
                          metric_params=metric_params, n_jobs=n_jobs)
        self.weights = _check_weights(weights)
        self.max_iter = max_iter
        self.fit = False
       
    def _fit(self, X):
        if self.metric_params is None:
            self.effective_metric_params_ = {}
        else:
            self.effective_metric_params_ = self.metric_params.copy()

        effective_p = self.effective_metric_params_.get('p', self.p)
        if self.metric in ['wminkowski', 'minkowski']:
            self.effective_metric_params_['p'] = effective_p

        self.effective_metric_ = self.metric
        # For minkowski distance, use more efficient methods where available
        if self.metric == 'minkowski':
            p = self.effective_metric_params_.pop('p', 2)
            if p < 1:
                raise ValueError("p must be greater than one "
                                 "for minkowski metric")
            elif p == 1:
                self.effective_metric_ = 'manhattan'
            elif p == 2:
                self.effective_metric_ = 'euclidean'
            elif p == np.inf:
                self.effective_metric_ = 'chebyshev'
            else:
                self.effective_metric_params_['p'] = p

        if isinstance(X, NeighborsBase):
            self._fit_X = X._fit_X
            self._tree = X._tree
            self._fit_method = X._fit_method
            return self

        elif isinstance(X, BallTree):
            self._fit_X = X.data
            self._tree = X
            self._fit_method = 'ball_tree'
            return self

        elif isinstance(X, KDTree):
            self._fit_X = X.data
            self._tree = X
            self._fit_method = 'kd_tree'
            return self

        X = check_array(X, accept_sparse='csr')

        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError("n_samples must be greater than 0")

        if issparse(X):
            if self.algorithm not in ('auto', 'brute'):
                warnings.warn("cannot use tree with sparse input: "
                              "using brute force")
            if self.effective_metric_ not in VALID_METRICS_SPARSE['brute']:
                raise ValueError("metric '%s' not valid for sparse input"
                                 % self.effective_metric_)
            self._fit_X = X.copy()
            self._tree = None
            self._fit_method = 'brute'
            return self

        self._fit_method = self.algorithm
        self._fit_X = X

        if self._fit_method == 'auto':
            # A tree approach is better for small number of neighbors,
            # and KDTree is generally faster when available
            if ((self.n_neighbors is None or
                 self.n_neighbors < self._fit_X.shape[0] // 2) and
                    self.metric != 'precomputed'):
                if self.effective_metric_ in VALID_METRICS['kd_tree']:
                    self._fit_method = 'kd_tree'
                else:
                    self._fit_method = 'ball_tree'
            else:
                self._fit_method = 'brute'

        if self._fit_method == 'ball_tree':
            self._tree = BallTree(X, self.leaf_size,
                                  metric=self.effective_metric_,
                                  **self.effective_metric_params_)
        elif self._fit_method == 'kd_tree':
            self._tree = KDTree(X, self.leaf_size,
                                metric=self.effective_metric_,
                                **self.effective_metric_params_)
        elif self._fit_method == 'brute':
            self._tree = None
        else:
            raise ValueError("algorithm = '%s' not recognized"
                             % self.algorithm)

        if self.n_neighbors is not None:
            if self.n_neighbors <= 0:
                raise ValueError(
                    "Expected n_neighbors > 0. Got %d" %
                    self.n_neighbors
                )

        return self
    
    def fit(self, X, y):
        """Fit the model using X as training data and y as target values
        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree}
            Training data. If array or matrix, shape [n_samples, n_features],
            or [n_samples, n_samples] if metric='precomputed'.
        y : {array-like, sparse matrix}
            Target values, array of float values, shape = [n_samples]
             or [n_samples, n_outputs]
        """
        if not isinstance(X, (KDTree, BallTree)):
            X, y = check_X_y(X, y, "csr", multi_output=True)
        self._y = y
        return self._fit(X)
    
    def _predict(self, X, y):
        """Predict the target for the provided data
        Parameters
        ----------
        X : array-like, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Test samples.
        Returns
        -------
        y : array of int, shape = [n_samples] or [n_samples, n_outputs]
            Target values
        """
        X = check_array(X, accept_sparse='csr')

        neigh_dist, neigh_ind = self.kneighbors(X)

        weights = _get_weights(neigh_dist, self.weights)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        if weights is None:
            y_pred = np.mean(_y[neigh_ind], axis=1)
        else:
            y_pred = np.empty((X.shape[0], _y.shape[1]), dtype=np.float64)
            denom = np.sum(weights, axis=1)

            for j in range(_y.shape[1]):
                num = np.sum(_y[neigh_ind, j] * weights, axis=1)
                y_pred[:, j] = num / denom

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred

    def predict(self,X):
        """Imputes the missing values in the data
        Parameters
        ----------
        X : array-like, shape (n_query, n_query), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Test samples.
        Returns
        -------
        y : array of int, shape = [n_query] or [n_query, n_query]
            Target values
        """
        self.X = X.copy()
        self.row_na_ = np.isnan(X).any(axis = 1) # True if theres atleast one NA in a row
        self.col_na_ = np.isnan(X).any(axis = 0) # True if theres atleast one NA in a col
        self.mat_na_ = np.isnan(X) 
        self._cols_ = range(X.shape[1])
        self._X_ = X[~row_na_] # complete data without any missing value
        # self.fit = True
        
        for _col_ in self._cols_:
            # perform this operation only if the col has missing values
            if self.col_na_[_col_]:
                # next two lines for selecting columns
                col_ind = np.ones(self.X.shape[1], dtype = bool)
                col_ind[_col_] = False
                # select all rows for which we have complete data of predictor variables 
                # but dont have data of target variable (missing y but all X)
                row_na = np.logical_and(~np.isnan(self.X[:,col_ind]).any(axis = 1), 
                                    np.isnan(self.X[:,_col_])) 
                # select indices of rows for which none of the data is missing
                # cannot use self.mat_na_ as I am goin to update the input in this loop
                # the idea is to use already imputed values to impute the to be imputed 
                # variables. As we go through this loop, the last variable to be impued
                # has the (dis)advantage of being imputed by the maximum number of 
                # observations
                miss_ind = np.isnan(self.X).any(axis=1)
                _ = self.fit(self.X[~miss_ind,col_ind], self.X[~miss_ind,_col_])
                # replace the missing values with the predictions
                self.X[row_na, _col_] = self._predict(self.X[row_na][:,col_ind])
        
        # perform this operation `self.max_iter` times
        for _iter_ in range(self.max_iter):
            for _col_ in self._cols_:
                # perform this operation only if the col has missing values
                if self.col_na_[_col_]:
                    # next two lines for selecting columns
                    col_ind = np.ones(self.X.shape[1], dtype = bool)
                    col_ind[_col_] = False
                    
                    # the values that had been imputed in the above for loop block are 
                    # being replaced with NaN. The idea is that the first variable to
                    # be imputed uses the least number of observations and so we perform
                    # the operation max_iter times each time assigning the imputed values
                    # to NaN and recomputing their weights.
                    
                    self.X[self.mat_na_[:,_col_],_col_] = np.nan
                    fit_ind = ~self.mat_na_[:,_col_]
                    _ = self.fit(self.X[fit_ind,:][:,col_ind], self.X[fit_ind,_col_])
                    self.X[self.mat_na_[:,_col_],_col_] = \
                    self._predict(self.X[self.mat_na_[:,col],:][:,col_ind])
        
        return self.X
