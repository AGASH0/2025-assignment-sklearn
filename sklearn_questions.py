"""Assignment - making a sklearn estimator and cv splitter.

The goal of this assignment is to implement by yourself:

- a scikit-learn estimator for the KNearestNeighbors for classification
  tasks and check that it is working properly.
- a scikit-learn CV splitter where the splits are based on a Pandas
  DateTimeIndex.

Detailed instructions for question 1:
The nearest neighbor classifier predicts for a point X_i the target y_k of
the training sample X_k which is the closest to X_i. We measure proximity with
the Euclidean distance. The model will be evaluated with the accuracy (average
number of samples corectly classified). You need to implement the `fit`,
`predict` and `score` methods for this class. The code you write should pass
the test we implemented. You can run the tests by calling at the root of the
repo `pytest test_sklearn_questions.py`. Note that to be fully valid, a
scikit-learn estimator needs to check that the input given to `fit` and
`predict` are correct using the `validate_data, check_is_fitted` functions
imported in this file.
You can find more information on how they should be used in the following doc:
https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator.
Make sure to use them to pass `test_nearest_neighbor_check_estimator`.


Detailed instructions for question 2:
The data to split should contain the index or one column in
datatime format. Then the aim is to split the data between train and test
sets when for each pair of successive months, we learn on the first and
predict of the following. For example if you have data distributed from
november 2020 to march 2021, you have have 4 splits. The first split
will allow to learn on november data and predict on december data, the
second split to learn december and predict on january etc.

We also ask you to respect the pep8 convention: https://pep8.org. This will be
enforced with `flake8`. You can check that there is no flake8 errors by
calling `flake8` at the root of the repo.

Finally, you need to write docstrings for the methods you code and for the
class. The docstring will be checked using `pydocstyle` that you can also
call at the root of the repo.

Hints
-----
- You can use the function:

from sklearn.metrics.pairwise import pairwise_distances

to compute distances between 2 sets of samples.
"""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics import pairwise_distances


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """KNearestNeighbors classifier.

    This classifier is compatible with scikit-learn and passes check_estimator.
    """

    def __init__(self, n_neighbors=1):
        """Initialize the estimator."""
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values."""
        # Validation des entrées
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        # Stockage des attributs
        self.classes_ = np.unique(y)
        self.X_train_ = X
        self.y_train_ = y

        # Nécessaire pour check_estimator (validation des dimensions)
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """Predict the class labels for the provided data."""
        check_is_fitted(self)
        X = check_array(X)

        # Vérification explicite des dimensions pour check_estimator
        if X.shape[1] != self.n_features_in_:
            msg = (
                f"X has {X.shape[1]} features, but KNearestNeighbors "
                f"is expecting {self.n_features_in_} features as input."
            )
            raise ValueError(msg)

        # Calcul des distances
        distances = pairwise_distances(X, self.X_train_)

        # Indices des k plus proches voisins
        k = self.n_neighbors
        neighbors_indices = np.argsort(distances, axis=1)[:, :k]

        # Labels correspondants
        neighbor_labels = self.y_train_[neighbors_indices]

        # Vote majoritaire
        y_pred = np.array([
            self._most_common(row) for row in neighbor_labels
        ])

        return y_pred

    def _most_common(self, row):
        """Trouve la classe majoritaire."""
        values, counts = np.unique(row, return_counts=True)
        return values[np.argmax(counts)]

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels."""
        return np.mean(self.predict(X) == y)


class MonthlySplit(BaseCrossValidator):
    """Monthly split cross-validator.

    Splits data based on monthly intervals (Rolling Window).
    Train: Month M, Test: Month M+1.
    """

    def __init__(self, time_col='index'):
        """Initialize the splitter."""
        self.time_col = time_col

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splitting iterations in the cross-validator."""
        dates = self._get_dates(X)
        months = dates.dt.to_period('M').unique()
        return max(0, len(months) - 1)

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set."""
        dates = self._get_dates(X)
        n_samples = len(dates)
        indices = np.arange(n_samples)

        # Conversion en périodes mensuelles
        months_period = dates.dt.to_period('M')
        unique_months = np.sort(months_period.unique())

        # Rolling Window : Train sur mois i, Test sur mois i+1
        for i in range(len(unique_months) - 1):
            train_month = unique_months[i]
            test_month = unique_months[i + 1]

            train_mask = months_period == train_month
            test_mask = months_period == test_month

            train_idx = indices[train_mask]
            test_idx = indices[test_mask]

            yield train_idx, test_idx

    def _get_dates(self, X):
        """Extract the date column from X."""
        if self.time_col == 'index':
            try:
                dates = X.index
            except AttributeError:
                raise ValueError("X n'a pas d'index.")
        else:
            try:
                dates = X[self.time_col]
            except (KeyError, TypeError):
                raise ValueError(f"Colonne '{self.time_col}' introuvable.")

        if not pd.api.types.is_datetime64_any_dtype(dates):
            raise ValueError("La colonne doit être de type datetime.")

        return pd.Series(dates)
