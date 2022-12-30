import numpy as np
from collections import Counter
import random
import time

DOT_DATA = """
digraph Tree {
node [shape=box, style="filled, rounded", color="black", fontname="helvetica"] ;
edge [fontname="helvetica"] ;
}
"""
CLASSIFICATION_CRITERIONS = ['gini', 'entropy']
REGRESSION_CRITERIONS = ['variance']


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


def calculate_variance(y):
    variance = np.var(y)
    
    return variance


def random_array(start, end, num_elements):
    arr = np.random.random((num_elements,))
    arr = start + arr * (end-start)
    return arr


def feature_split_interval(X_feature, interval_granularity=0.5, default_max_len_intervals=3, extreme_random=False):
    """
    Get the intervals for a particular feature to find the gini impurity values for the split.
    """
    X_feature = X_feature.reshape(-1,)

    # print(X_feature.shape)

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
    elif unique_values.shape[0] == 3:
        # print(np.array([unique_values.mean()]))
        return np.array([unique_values[1]])


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
    tic = time.time()
    split_intervals = feature_split_interval(X_feature, extreme_random=extreme_random)
    time_taken = time.time() - tic

    # if criterion == 'gini':
    #     parent_node_criterion_value = calculate_gini_impurity(y)
    # elif criterion == 'entropy':
    #     parent_node_criterion_value = calculate_entropy(y)
    # elif criterion == 'variance':
    #     parent_node_criterion_value = calculate_variance(y)

    if criterion in CLASSIFICATION_CRITERIONS:
        min_weighted_criterion_value = 1
    else:
        min_weighted_criterion_value = float('inf')

    # print(split_intervals, X_feature)
    optimal_split_values = [split_intervals[0]]

    time_taken = 0   

    num_samples = len(y) 

    for split_value in split_intervals:
        # print(split_value)
        # tic = time.time()
        ge_split = y[X_feature >= split_value]
        lt_split = y[X_feature < split_value]

        w1 = len(lt_split)/num_samples
        w2 = len(ge_split)/num_samples

        if ge_split.shape[0] == 0 or lt_split.shape[0] == 0:
            continue

        # w1 = 0.35
        # w2 = 0.65
        
        if criterion == 'gini':
            tic = time.time()
            weighted_criterion_value = (calculate_gini_impurity(lt_split) * len(lt_split) \
                + calculate_gini_impurity(ge_split) * len(ge_split))/ num_samples
            toc = time.time()
        elif criterion == 'entropy':
            tic = time.time()
            weighted_criterion_value = (calculate_entropy(lt_split) * len(lt_split) \
                + calculate_entropy(ge_split) * len(ge_split))/ num_samples
            toc = time.time()
        elif criterion == 'variance':
            # weighted_criterion_value = (calculate_variance(lt_split) * len(lt_split) \
            #     + calculate_variance(ge_split) * len(ge_split))/ len(y)
            tic = time.time()
            var_lt_split = calculate_variance(lt_split)
            var_ge_split = calculate_variance(ge_split)
            toc = time.time()
            weighted_criterion_value = var_lt_split * w1 + var_ge_split * w2
            # toc = time.time()

        time_taken += toc-tic
        
        if weighted_criterion_value <= min_weighted_criterion_value:
            if weighted_criterion_value == min_weighted_criterion_value:
                optimal_split_values.append(split_value)
            else:
                optimal_split_values = [split_value]
                min_weighted_criterion_value = weighted_criterion_value
    
    return np.mean(optimal_split_values), min_weighted_criterion_value, time_taken


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


def colorhex(red, green, blue):
    return f"{int(red):02x}{int(green):02x}{int(blue):02x}"


def random_color(rgb_pick=0):
    # rgb_pick = random.randint(0,2)
    red = random.randint(0,96)
    green = random.randint(0,96)
    blue = random.randint(0,96)
    if rgb_pick == 0:
        red = random.randint(192,255)
    elif rgb_pick == 1:
        green = random.randint(192,255)
    else:
        blue = random.randint(192,255)
    return colorhex(red, green, blue)


def calculate_node_fillcolor(value, class_colors):
    total = sum(value)
    proportions = [v/total for v in value]
    pp = {cc: pr for cc,pr in zip(class_colors, proportions)}
    max_pr = max(proportions)
    proportions.remove(max_pr)
    second_max_pr = max(proportions)
    white_weight = 0.3/max(0.01,max_pr-second_max_pr)
    pp['ffffff'] = white_weight
    # print(pp)
    fillcolor = combine_hex_values(pp)
    return "#"+fillcolor


class Node:
    """
    A class to represent a node(leaf) of a decision tree.
    """

    def __init__(self, X, y, height, max_tree_height, criterion, extreme_random, min_samples_leaf):
        self.X = X
        self.y = y
        self.height = height
        self.max_tree_height = max_tree_height
        self.criterion = criterion
        self.extreme_random = extreme_random
        self.min_samples_leaf = min_samples_leaf
        if criterion == 'gini':
            self.criterion_value = calculate_gini_impurity(y)
        elif criterion == 'entropy': 
            self.criterion_value = calculate_entropy(y)
        elif criterion == 'variance': 
            if y.shape[0]==0:
                print(1)
            self.criterion_value = calculate_variance(y)
        
        self.num_samples, self.num_features = X.shape

        if criterion in CLASSIFICATION_CRITERIONS:
            self.node_value = Counter(y).most_common()[0][0]
        else:
            self.node_value = np.mean(y)
        self.left_child = None
        self.right_child = None

        # print(self.gini_impurity)

        if self.criterion_value == 0 or self.height >= max_tree_height or self.num_samples < min_samples_leaf:
            self.is_leaf = True
            # print(self.gini_impurity, self.height, self.max_tree_height)
            self.feature_split = None
            self.split_value = None
            self.split_criterion_value = 0
        else:
            self.is_leaf = False
            self.feature_split = random.choice([*range(self.num_features)])
            # print(X[:,self.feature_split])
            split_intervals = feature_split_interval(X[:,self.feature_split])
            # print(split_intervals)
            self.split_value = np.random.choice(split_intervals)
        
        # print(X.shape, y.shape, self.criterion_value, self.is_leaf)
        self.time_taken_for_split = 0


    def fit(self, features_to_skip=[]):
        # print(self.X.shape)
        if self.criterion in CLASSIFICATION_CRITERIONS:
            min_criterion_value = 1
        else:
            min_criterion_value = float('inf')

        total_time_taken = 0
        for i in range(self.num_features):
            if i in features_to_skip:
                continue
            # tic = time.time()
            split_value, split_criterion_value, time_taken = optimal_feature_split(self.X[:,i], self.y, criterion=self.criterion, extreme_random=self.extreme_random)
            # toc = time.time()
            # time_taken = toc-tic
            total_time_taken += time_taken
            if split_criterion_value < min_criterion_value:
                min_criterion_value = split_criterion_value
                best_feature = i
                best_feature_split_value = split_value
        
        X_feature = self.X[:, best_feature]
        
        X_lt = self.X[X_feature < best_feature_split_value, :]
        X_ge = self.X[X_feature >= best_feature_split_value, :]

        y_lt = self.y[X_feature < best_feature_split_value]
        y_ge = self.y[X_feature >= best_feature_split_value]

        # print(X_lt.shape, X_ge.shape)

        self.feature_split = best_feature
        self.split_value = best_feature_split_value
        self.split_criterion_value = min_criterion_value
        self.time_taken_for_split = total_time_taken

        # print(best_feature, best_feature_split_value, min_criterion_value, self.height)

        # print(self.criterion_value, min_criterion_value, self.criterion_value <= min_criterion_value)

        if self.criterion_value <= min_criterion_value:
            # This means the split did not improve the criteron - gini impurity, we need to stop
            return self

        self.left_child = Node(X_lt, y_lt, self.height+1, self.max_tree_height, self.criterion, self.extreme_random, self.min_samples_leaf)
        self.right_child = Node(X_ge, y_ge, self.height+1, self.max_tree_height, self.criterion, self.extreme_random, self.min_samples_leaf)

        if not self.left_child.is_leaf:
            self.left_child.fit(features_to_skip=features_to_skip)

        if not self.right_child.is_leaf:
            self.right_child.fit(features_to_skip=features_to_skip)

        return self


    def predict_sample(self, X_sample):
        if self.is_leaf:
            return self.node_value
        if X_sample[self.feature_split] >= self.split_value:
            # Pass on to the right child
            if self.right_child is not None:
                return self.right_child.predict_sample(X_sample)
            else:
                return self.node_value
        else:
            if self.left_child is not None:
                return self.left_child.predict_sample(X_sample)
            else:
                return self.node_value


    def total_time_taken(self):
        time_taken = self.time_taken_for_split
        if self.is_leaf:
            return time_taken
        if self.right_child is not None:
            time_taken += self.right_child.total_time_taken()
        if self.left_child is not None:
            time_taken += self.left_child.total_time_taken()
        return time_taken

    
    def export_graphviz_node(self, 
            dot_data, 
            num_nodes, 
            parent_node_num, 
            num_classes, 
            feature_names=None, 
            class_names=None, 
            class_colors=None,
            regression=False,
            ):
        current_node_num = num_nodes
        num_nodes += 1
        label_text = f"{current_node_num}"
        # self.feature_split = best_feature
        # self.split_value = best_feature_split_value
        # self.split_criterion_value = min_criterion_value

        if class_names is not None:
            node_value = class_names[self.node_value]
        else:
            node_value = self.node_value

        counts = Counter(self.y)
        # value = "["
        # for nc in range(num_classes-1):
        #     value += str(counts.get(nc, 0)) + ', '
        # value += str(counts.get(num_classes-1, 0)) + ']'

        value = [counts.get(nc, 0) for nc in range(num_classes)]

        if not regression:
            value_text = f"value = {value}<br/>class"
        else:
            value_text = f"value"

        if self.feature_split is not None:
            if feature_names is not None:
                feature_split = feature_names[self.feature_split]
            else:
                feature_split = str(self.feature_split)
            label_text = f"{feature_split} &lt; {self.split_value:.2f}<br/>{self.criterion} = {self.criterion_value:.2f}<br/>samples = {self.num_samples}<br/>{value_text} = {node_value}"
        else:
            label_text = f"{self.criterion} = {self.split_criterion_value}<br/>samples = {self.num_samples}<br/>{value_text} = {node_value}"
        
        fillcolor = '#fffdfc'
        if class_colors is not None:
            fillcolor = calculate_node_fillcolor(value, class_colors)
        dot_data = add_node(dot_data, current_node_num, parent_node_num, label_text, fillcolor)
        if self.is_leaf:
            return dot_data, num_nodes
        
        if self.left_child is not None:
            dot_data, num_nodes = self.left_child.export_graphviz_node(dot_data, num_nodes, current_node_num, \
                num_classes, feature_names=feature_names, class_names=class_names, class_colors=class_colors, \
                regression=regression)
        if self.right_child is not None:
            dot_data, num_nodes = self.right_child.export_graphviz_node(dot_data, num_nodes, current_node_num, \
                num_classes, feature_names=feature_names, class_names=class_names, class_colors=class_colors, \
                regression=regression)

        return dot_data, num_nodes


class DecisionTree:
    """
    A Decision Tree Machine Learning algorithm.

    Parameters:
        max_depth (int): The maximum depth of the decision tree
        criterion (str): The criterion for splitting - 'gini', 'entropy' or 'variance'
        extreme_random (bool): Whether to use randomly generated thresholds for splitting (default=False)
        features_to_skip (list): The list of feature numbers to not consider for building the decision tree
    """

    def __init__(self, 
        max_depth: int=20, 
        criterion: str='gini', 
        min_samples_leaf: int=5,
        extreme_random: bool=False,
        features_to_skip: list=[]
        ):
        self.max_depth = max_depth
        self.criterion = criterion
        self.extreme_random = extreme_random
        self.features_to_skip = features_to_skip
        self.min_samples_leaf = min_samples_leaf


    def fit(self, X: np.array, y: np.array):
        n_samples, n_features = X.shape
        assert n_samples == y.shape[0]
        assert len(y.shape) == 1 or y.shape[1] == 1

        self.num_classes = len(Counter(y))

        self.root_node = Node(X, y, 1, self.max_depth, self.criterion, self.extreme_random, self.min_samples_leaf)
        self.root_node.fit(features_to_skip=self.features_to_skip)

        return self

    
    def predict(self, X):
        y_pred = np.array([self.root_node.predict_sample(X_sample) for X_sample in X])
        return y_pred


    def total_time_taken(self):
        return self.root_node.total_time_taken()


    def export_graphviz(self, feature_names, class_names=None, regression=False):       
        class_colors = None
        if not regression:
            class_colors = [random_color(i%3) for i in range(self.num_classes)]
        dot_data, _ = self.root_node.export_graphviz_node(DOT_DATA, 0, -1, self.num_classes, \
            feature_names=feature_names, class_names=class_names, class_colors=class_colors,\
            regression=regression)
        return dot_data


