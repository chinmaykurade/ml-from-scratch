import numpy as np
from collections import Counter
import random
from tree.decision_tree import DecisionTree


class RandomForestBase:
    """
    A Random Forest Classifier Machine Learning algorithm.

    Parameters:
        n_estimators (int): The number of decision trees to use
        max_tree_depth (int): The maximum depth of the component decision tree
        criterion (str): The criterion for splitting - 'gini' or 'entropy'
        max_features_split (float): The maximum number of features to consider for making a decision tree split (default=number of features)
        extreme_random (bool): Whether to use randomly generated thresholds for splitting (default=False)
    """

    def __init__(self, 
        n_estimators: int=100, 
        max_tree_depth: int=20, 
        criterion: str='gini',
        max_features_split: int=None, 
        min_samples_leaf: int=5,
        extreme_random: bool=False
        ):
        self.n_estimators = n_estimators
        self.max_tree_depth = max_tree_depth
        self.criterion = criterion
        self.max_features_split = max_features_split
        self.extreme_random = extreme_random
        self.min_samples_leaf = min_samples_leaf


    def generate_random_decision_trees(self, X, y, n_estimators, max_features_split, random_state=100):
        # print(n_estimators)
        estimators = []
        for _ in range(n_estimators):
            features_to_skip = []
            if self.max_features_split is not None:
                # print(self.n_features, self.n_features-self.max_features_split)
                features_to_skip = list(np.random.choice(self.n_features, self.n_features-self.max_features_split, replace=False))
                # print(features_to_skip)

            clf = DecisionTree(
                max_depth=self.max_tree_depth, 
                criterion=self.criterion, 
                features_to_skip=features_to_skip,
                extreme_random=self.extreme_random,
                min_samples_leaf=self.min_samples_leaf
                )
            random_indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
            X_sample = X[random_indices]
            y_sample = y[random_indices]
            clf.fit(X_sample, y_sample)
            estimators.append(clf)
        return estimators


    def time_taken_train(self):
        return sum([c.total_time_taken() for c in self.estimators])



class RandomForestClassifier(RandomForestBase):
    def __init__(self, 
        n_estimators: int=100, 
        max_tree_depth: int=20, 
        criterion: str='gini',
        max_features_split: int=None,
        min_samples_leaf: int=5,
        extreme_random: bool=False
        ):
        super().__init__(n_estimators=n_estimators, max_tree_depth=max_tree_depth, 
            criterion=criterion, max_features_split=max_features_split,
            extreme_random=extreme_random, min_samples_leaf=min_samples_leaf
            )


    def fit(self, X: np.array, y: np.array):
        n_samples, n_features = X.shape
        assert n_samples == y.shape[0]
        assert len(y.shape) == 1 or y.shape[1] == 1

        self.n_samples, self.n_features = n_samples, n_features

        self.num_classes = len(Counter(y))

        self.estimators = self.generate_random_decision_trees(X, y, self.n_estimators, self.max_features_split)

        return self


    def predict_proba(self, X):
        y_preds = [clf.predict(X) for clf in self.estimators]
        y_preds = [np.squeeze(np.eye(self.num_classes)[a.reshape(-1)]) for a in y_preds]
        y_pred_proba = np.array(y_preds).mean(axis=0)

        return y_pred_proba


    def predict(self, X):
        y_pred_proba = self.predict_proba(X)
        y_pred = y_pred_proba.argmax(axis=1)

        return y_pred


class RandomForestRegressor(RandomForestBase):
    def __init__(self, 
        n_estimators: int=100, 
        max_tree_depth: int=20, 
        criterion: str='variance',
        max_features_split: int=None,
        min_samples_leaf: int=5,
        extreme_random: bool=False
        ):
        super().__init__(n_estimators=n_estimators, max_tree_depth=max_tree_depth, 
            criterion=criterion, max_features_split=max_features_split,
            extreme_random=extreme_random, min_samples_leaf=min_samples_leaf
            )


    def fit(self, X: np.array, y: np.array):
        n_samples, n_features = X.shape
        assert n_samples == y.shape[0]
        assert len(y.shape) == 1 or y.shape[1] == 1

        self.n_samples, self.n_features = n_samples, n_features

        self.estimators = self.generate_random_decision_trees(X, y, self.n_estimators, self.max_features_split)

        return self


    def predict(self, X):
        y_preds = np.array([clf.predict(X) for clf in self.estimators]).mean(axis=0)

        return y_preds