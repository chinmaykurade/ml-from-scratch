import numpy as np
from collections import Counter
import random

def calculate_gini_impurity(y):
    gini = 0
    total = len(y)

    n_classes = len(set(y))
    counts = Counter(y)
    
    for k,v in counts.items():
        gini += (v/total)**2
    gini_impurity = 1 - gini
    
    return gini_impurity


def calculate_entropy(y):
    entropy = 0
    total = len(y)

    n_classes = len(set(y))
    counts = Counter(y)
    
    for k,v in counts.items():
        p_class = v/total
        # print(p_class)
        entropy += -(p_class) * np.log(p_class) / np.log(2)
    
    return entropy


def random_array(start, end, num_elements):
    arr = np.random.random((num_elements,))
    arr = start + arr * (end-start)
    return arr


def feature_split_interval(X_feature, interval_granularity=0.2, default_max_len_intervals=5, extreme_random=False):
    """
    Get the intervals for a particular feature to find the gini impurity values for the split.
    """
    X_feature = X_feature.reshape(-1,)

    feature_max, feature_min = max(X_feature), min(X_feature)

    max_len_intervals = max(default_max_len_intervals, int(interval_granularity * X_feature.shape[0]))
    # print(max_len_intervals)

    if extreme_random:
        return random_array(feature_min, feature_max, max_len_intervals)

    X_feature_sorted = np.sort(X_feature)

    unique_values = np.unique(X_feature)

    if unique_values.shape[0] == 1:
        # print(np.array([unique_values[0]]))
        return np.array([unique_values[0]])
    elif unique_values.shape[0] == 2:
        # print(np.array([unique_values.mean()]))
        return np.array([unique_values.mean()])

    # return X_feature_sorted

    cum_diff = np.roll(X_feature_sorted, -1) - X_feature_sorted
    cum_diff = cum_diff[:-1]
    # max_diff, min_diff = max(cum_diff), min(cum_diff)
    # median_diff = np.median(cum_diff)
    # median_nonzero_diff = np.median(cum_diff[cum_diff!=0])
    min_nonzero_diff = np.min(cum_diff[cum_diff!=0])

    start = feature_min + min_nonzero_diff/2
    end = feature_max - min_nonzero_diff/2
    min_nonzero_diff_interval = np.arange(start, end, min_nonzero_diff)

    # print(start, end, min_nonzero_diff_interval)

    if min_nonzero_diff_interval.shape[0] > max_len_intervals:
        return np.linspace(start, end, max_len_intervals)

    return min_nonzero_diff_interval


def optimal_feature_split(X_feature, y, criterion='gini', extreme_random=False):
    split_intervals = feature_split_interval(X_feature, extreme_random=extreme_random)

    if criterion == 'gini':
        parent_node_criterion_value = calculate_gini_impurity(y)
    else:
        parent_node_criterion_value = calculate_entropy(y)

    min_weighted_criterion_value = 1

    # print(split_intervals, X_feature)
    optimal_split_values = [split_intervals[0]]

    for split_value in split_intervals:
        # print(split_value)
        ge_split = y[X_feature >= split_value]
        lt_split = y[X_feature < split_value]

        if criterion == 'gini':
            weighted_criterion_value = (calculate_gini_impurity(lt_split) * len(lt_split) \
                + calculate_gini_impurity(ge_split) * len(ge_split))/ len(y)
        else:
            weighted_criterion_value = (calculate_entropy(lt_split) * len(lt_split) \
                + calculate_entropy(ge_split) * len(ge_split))/ len(y)
        
        if weighted_criterion_value <= min_weighted_criterion_value:
            if weighted_criterion_value == min_weighted_criterion_value:
                optimal_split_values.append(split_value)
            else:
                optimal_split_values = [split_value]
                min_weighted_criterion_value = weighted_criterion_value
    
    return np.mean(optimal_split_values), min_weighted_criterion_value


def add_node(dot_data, num_nodes, parent_node_num, label_text, fillcolor):
    current_node_num = num_nodes
    dot_data = dot_data.split('}')[0]
    dot_data += f'{current_node_num} [label=<{label_text}>, fillcolor="{fillcolor}"] ;'
    dot_data += '\n'
    if parent_node_num >= 0:
        dot_data += f'{parent_node_num} -> {current_node_num}'
        dot_data += '\n'
    dot_data += '}'
    return dot_data


def combine_hex_values(d):
    d_items = sorted(d.items())
    tot_weight = sum(d.values())
    red = int(sum([int(k[:2], 16)*v for k, v in d_items])/tot_weight)
    green = int(sum([int(k[2:4], 16)*v for k, v in d_items])/tot_weight)
    blue = int(sum([int(k[4:6], 16)*v for k, v in d_items])/tot_weight)
    zpad = lambda x: x if len(x)==2 else '0' + x
    return zpad(hex(red)[2:]) + zpad(hex(green)[2:]) + zpad(hex(blue)[2:])


def calculate_node_fillcolor(value, class_colors):
    total = sum(value)
    proportions = [v/total for v in value]
    pp = {cc: pr for cc,pr in zip(class_colors, proportions)}
    print(pp)
    fillcolor = combine_hex_values(pp)
    return "#"+fillcolor


class Node:
    """
    A class to represent a node(leaf) of a decision tree.
    """

    def __init__(self, X, y, height, max_tree_height, criterion, extreme_random):
        self.X = X
        self.y = y
        self.height = height
        self.max_tree_height = max_tree_height
        self.criterion = criterion
        self.extreme_random = extreme_random
        if criterion == 'gini':
            self.criterion_value = calculate_gini_impurity(y)
        else: 
            self.criterion_value = calculate_entropy(y)
        
        self.num_samples, self.num_features = X.shape
        self.final_class = Counter(y).most_common()[0][0]
        self.left_child = None
        self.right_child = None

        # print(self.gini_impurity)

        if self.criterion_value == 0 or self.height >= max_tree_height:
            self.is_terminal = True
            # print(self.gini_impurity, self.height, self.max_tree_height)
            self.feature_split = None
            self.split_value = None
            self.split_criterion_value = 0
        else:
            self.is_terminal = False
            self.feature_split = random.choice([*range(self.num_features)])
            # print(X[:,self.feature_split])
            split_intervals = feature_split_interval(X[:,self.feature_split])
            # print(split_intervals)
            self.split_value = np.random.choice(split_intervals)
        
        # print(X.shape, y.shape, self.criterion_value, self.is_terminal)


    def fit(self):
        # print(self.X.shape)
        min_criterion_value = 1
        for i in range(self.num_features):
            split_value, split_criterion_value = optimal_feature_split(self.X[:,i], self.y, criterion=self.criterion, extreme_random=self.extreme_random)
            if split_criterion_value < min_criterion_value:
                min_criterion_value = split_criterion_value
                best_feature = i
                best_feature_split_value = split_value
        
        X_feature = self.X[:, best_feature]
        
        X_lt = self.X[X_feature < best_feature_split_value, :]
        X_ge = self.X[X_feature >= best_feature_split_value, :]

        y_lt = self.y[X_feature < best_feature_split_value]
        y_ge = self.y[X_feature >= best_feature_split_value]

        self.feature_split = best_feature
        self.split_value = best_feature_split_value
        self.split_criterion_value = min_criterion_value

        # print(best_feature, best_feature_split_value, min_gini_impurity, self.height)

        if self.criterion_value <= min_criterion_value:
            # This means the split did not improve the criteron - gini impurity, we need to stop
            return self

        self.left_child = Node(X_lt, y_lt, self.height+1, self.max_tree_height, self.criterion, self.extreme_random)
        self.right_child = Node(X_ge, y_ge, self.height+1, self.max_tree_height, self.criterion, self.extreme_random)

        if not self.left_child.is_terminal:
            self.left_child.fit()

        if not self.right_child.is_terminal:
            self.right_child.fit()

        return self


    def predict_sample(self, X_sample):
        if self.is_terminal:
            return self.final_class
        if X_sample[self.feature_split] >= self.split_value:
            # Pass on to the right child
            if self.right_child is not None:
                return self.right_child.predict_sample(X_sample)
            else:
                return self.final_class
        else:
            if self.left_child is not None:
                return self.left_child.predict_sample(X_sample)
            else:
                return self.final_class

    
    def export_graphviz_node(self, dot_data, num_nodes, parent_node_num, num_classes, feature_names=None, class_names=None, class_colors=None):
        current_node_num = num_nodes
        num_nodes += 1
        label_text = f"{current_node_num}"
        # self.feature_split = best_feature
        # self.split_value = best_feature_split_value
        # self.split_criterion_value = min_criterion_value

        if class_names is not None:
            final_class = class_names[self.final_class]
        else:
            final_class = self.final_class

        counts = Counter(self.y)
        # value = "["
        # for nc in range(num_classes-1):
        #     value += str(counts.get(nc, 0)) + ', '
        # value += str(counts.get(num_classes-1, 0)) + ']'

        value = [counts.get(nc, 0) for nc in range(num_classes)]

        if self.feature_split is not None:
            if feature_names is not None:
                feature_split = feature_names[self.feature_split]
            else:
                feature_split = str(self.feature_split)
            label_text = f"{feature_split} &lt; {self.split_value:.2f}<br/>{self.criterion} = {self.split_criterion_value:.2f}<br/>samples = {self.num_samples}<br/>value = {value}<br/>class = {final_class}"
        else:
            label_text = f"{self.criterion} = {self.split_criterion_value}<br/>samples = {self.num_samples}<br/>value = {value}<br/>class = {final_class}"
        
        fillcolor = '#fffdfc'
        if class_colors is not None:
            fillcolor = calculate_node_fillcolor(value, class_colors)
        dot_data = add_node(dot_data, current_node_num, parent_node_num, label_text, fillcolor)
        if self.is_terminal:
            return dot_data, num_nodes
        
        if self.left_child is not None:
            dot_data, num_nodes = self.left_child.export_graphviz_node(dot_data, num_nodes, current_node_num, num_classes, feature_names=feature_names, class_names=class_names, class_colors=class_colors)
        if self.right_child is not None:
            dot_data, num_nodes = self.right_child.export_graphviz_node(dot_data, num_nodes, current_node_num, num_classes, feature_names=feature_names, class_names=class_names, class_colors=class_colors)

        return dot_data, num_nodes


class DecisionTreeClassifier:
    """
    A Decision Tree Classifier Machine Learning algorithm.

    Parameters:
        max_depth (int): The maximum depth of the decision tree
        criterion (str): The criterion for splitting - 'gini' or 'information gain'
        extreme_random (bool): Whether to use randomly generated thresholds for splitting (default=False)
    """

    def __init__(self, max_depth: int=20, criterion: str='gini', extreme_random: bool=False):
        self.max_depth = max_depth
        self.criterion = criterion
        self.extreme_random = extreme_random


    def fit(self, X: np.array, y: np.array):
        n_samples, n_features = X.shape
        assert n_samples == y.shape[0]
        assert len(y.shape) == 1 or y.shape[1] == 1

        self.num_classes = len(Counter(y))

        self.root_node = Node(X, y, 1, self.max_depth, self.criterion, self.extreme_random)
        self.root_node.fit()

        return self

    
    def predict(self, X):
        y_pred = np.array([self.root_node.predict_sample(X_sample) for X_sample in X])
        return y_pred


    def export_graphviz(self, feature_names, class_names):
        dot_data = """
digraph Tree {
node [shape=box, style="filled, rounded", color="black", fontname="helvetica"] ;
edge [fontname="helvetica"] ;
}
"""
        
        class_colors = ["e58139", "39e581", "8139e5"]
        dot_data, _ = self.root_node.export_graphviz_node(dot_data, 0, -1, self.num_classes, feature_names=feature_names, class_names=class_names, class_colors=class_colors)
        return dot_data