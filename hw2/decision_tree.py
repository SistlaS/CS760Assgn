import pandas as pd
import numpy as np
import argparse

#https://www.kaggle.com/code/fareselmenshawii/decision-tree-from-scratch

class Node:
    def __init__(self, feature=None, gain=None, left=None, right=None, value=None) -> None:
        self.feature = feature
        self.gain = gain
        self.left_node = left
        self.right_node = right
        self.value = value
    def print_node(self):
        print("Node feature ", self.feature)
        print("Node gain ", self.gain)
        print("Node value ", self.value)

class DecisionTree:
    def __init__(self) -> None:
        self.depth = 0
        self.features = list
        self.x_train = np.array
        self.y_train = np.array
    
    def entropy(self, y):
        """
        computes entropy of given labels

        params:
            y(ndarray) : Input label values
        outputs:
            entropy(float) : entropy of given label values
        """
        
        entropy = 0
        labels = np.unique(y)
        for label in labels:
            label_counts = y[y == label]
            p1 = len(label_counts) / len(y)
            entropy += -p1*np.log2(p1)

        return entropy


    def calc_info_gain(self, parent, left, right):
        info_gain = 0
        # print("calc parent entropy ", parent)
        parent_entropy = self.entropy(parent)

        left_size = len(left)/len(parent)
        right_size = len(right)/len(parent)

        left_entropy, right_entropy = self.entropy(left), self.entropy(right)

        weighted_entropy = left_size*left_entropy + right_size*right_entropy
        info_gain = parent_entropy - weighted_entropy

        return info_gain


    def split_dataset(self, dataset, feature, threshold):
        """"
        splits the dataset into two datasets based on the given feature and threshold
        params:

        returns:
            left_dataset:
            right_dataset:
        """
        left = []
        right = []

        for row in dataset:
            if row[feature] <= threshold:
                left.append(row)
            else:
                right.append(row)
        
        left = np.array(left)
        right = np.array(right)

        return left, right

    def best_split(self, dataset, n_samples, n_features):
        """
        finds the best split for the given dataset
        returs a dict: with best split feature index, threshold, gain, left & right datasets
        """

        best_split = {
            "gain" : -1,
            "threshold" : None,
            "feature" : None
        }

        for feature_index in range(n_features):
            feature_values = dataset[:, feature_index]

            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                left_dataset, right_dataset = self.split_dataset(dataset, feature_index, threshold)

                if len(left_dataset) and len(right_dataset):
                    #get the y values of parent, left and right node to calculate info gain
                    y, left_y, right_y = dataset[:,-1], left_dataset[:,-1], right_dataset[:,-1]
                    info_gain = self.calc_info_gain(y, left_y, right_y)

                    #update the best split based on the info gain
                    if info_gain > best_split.get("gain", -1):
                        best_split["gain"] = info_gain
                        best_split["threshold"] = threshold
                        best_split["feature"] = feature_index
                        best_split["left_dataset"] = left_dataset
                        best_split["right_dataset"] = right_dataset
        
        return best_split

    def calculate_leaf_value(self, y):
        """
        calculates the most occuring value of y in the given list of y
        """

        y = list(y)

        count_0 = y.count(0.0)
        count_1 = y.count(1.0)

        if count_1 >= count_0:
            return 1.0
        else:
            return 0.0
    
    def build_tree(self, dataset, current_depth=0):
        X, y = dataset[:, :-1], dataset[:,-1]
        n_samples, n_features = X.shape

        if n_samples == 0:
            print("# samples = 0 : stopping and returning None")
            return None
        
        current_entropy = self.entropy(y)
        
        if current_entropy == 0:
            print("current_entropy = 0 : stopping and returning leaf node :", self.calculate_leaf_value(y))
            return Node(value=self.calculate_leaf_value(y))
        
        #keep splitting the dataset until the stopping criteria is met
        best_split = self.best_split(dataset, n_samples, n_features)
        
        if best_split["gain"] == 0:
            print("gain is 0 : stopping and returning leaf node :", self.calculate_leaf_value(y))
            return Node(value=self.calculate_leaf_value(y))
        else:
            left_node = self.build_tree(best_split["left_dataset"], current_depth+1)
            right_node = self.build_tree(best_split["right_dataset"], current_depth+1)

            return Node(best_split["feature"], best_split["gain"], left_node, right_node, best_split["gain"])


    def fit(self, X, y):
        dataset = np.concatenate((X, y), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        """
        predicts the output for each instance in the feature matrix X
        """

        predictions = []

        for each in X:
            prediction = self.make_prediction(each, self.root)
            predictions.append(prediction)

        return np.array(predictions)
        
    def make_prediction(self, x, node):
        print(x, node.feature, node.value)
        if node.value != None:
            print(node.value, "leaf node value")
            return node.value
        else:
            if x[node.feature] <= node.threshold:
                return self.make_prediction(x, node.left)
            else:
                return self.make_prediction(x, node.right)

def read_data(filename):
    data = pd.read_csv(filename, sep= ' ', index_col=False, header=None, names = ['x1', 'x2', 'y'])
    X = data[['x1','x2']].values
    y = data[['y']].values

    return X, y

def split_train_test(X, y, random_state=41, test_size=0.2):
    n_samples = X.shape[0]
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(np.arange(n_samples))

    test_size = int(n_samples*test_size)

    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

def accuracy(y_true, y_pred):
    """
    Computes the accuracy of a classification model.

    Parameters:
    ----------
        y_true (numpy array): A numpy array of true labels for each data point.
        y_pred (numpy array): A numpy array of predicted labels for each data point.

    Returns:
    ----------
        float: The accuracy of the model
    """
    y_true = y_true.flatten()
    total_samples = len(y_true)
    correct_predictions = np.sum(y_true == y_pred)
    return (correct_predictions / total_samples) 

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='take input data filename')
    parser.add_argument('--filename', metavar='f', type=str)
    args = parser.parse_args()
    filename = args.filename
    
    X, y = read_data(filename)
    X_train, X_test, y_train, y_test = split_train_test(X, y, random_state=41, test_size=0.2)


    model = DecisionTree()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(X_test, " : ", predictions)
    print(f"Model's Accuracy: {accuracy(y_test, predictions)}")