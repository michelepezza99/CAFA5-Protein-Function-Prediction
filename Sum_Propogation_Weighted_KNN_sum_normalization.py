import numpy as np
import pandas as pd
import sklearn
from skmultilearn.adapt import MLkNN
from sklearn.neighbors import KNeighborsClassifier
import json

X_train = pd.read_csv(r"C:\Users\hb102\Documents\Python\Train\X_train_principal.csv", header = 0).to_numpy()

Y_train = pd.read_csv(r"C:\Users\hb102\Documents\Python\Train\y_train.csv", header = 0).to_numpy()

X_val = pd.read_csv(r"C:\Users\hb102\Documents\Python\Train\X_val_principal.csv", header = 0).to_numpy()
Y_val = pd.read_csv(r"C:\Users\hb102\Documents\Python\Train\y_val.csv", header = 0).to_numpy()

info_content = pd.read_csv(r"C:\Users\hb102\Documents\Python\Train\information_content.csv", header=0).to_numpy().reshape(1,1500)


label_matrix = pd.read_csv(r"C:\Users\hb102\OneDrive\Desktop\CAFA-5\Train\label_matrix_1500.csv", nrows=0)
labels = label_matrix.columns.tolist()[1:]


with open('ancestor_dict.json', 'r') as f:
    ancestor_dict = json.load(f)


def prop_labels(labels, k, neighbor_labels, ancestor_dict, info_content, weights, y_pred_scores, a):
    """
    Voting algorithm that adds (info-accretion * inverse cosine distance) weighted votes for ancestors of relevant GO labels 

    Args:
        labels: list of GO annotations in order they occur in ancestor_dict, info_content, and binary label matrices
        k: number of nearest neighbors in KNN search
        neighbor_labels: ndarray of shape (k, n_features or 1500) with each nearest neighbor's binary labels
        ancestor_dict: dictionary of ancestors for top 1500 GO annotations
        info_content: ndarray of shape (1, n_features or 1500) of info content values for top 1500 GO annotations
        weights: ndarray of shape (1, k) with inverse cos distance weights of k neighbors from test datapoint
        y_pred_scores: ndarray of shape (# of test datapoints, n_features or 1500) where votes are accumulated
        a: integer corresponding to row of y_pred_scores being updated

    Returns:
        None
    """
    if len(labels) != info_content.shape[1]:
        raise ValueError("Length of labels does not match info_content columns")

    for i in range(k):
        single_label = neighbor_labels[i]
        
        for j in range(neighbor_labels.shape[1]):
            if single_label[j] == 1:
                ancestors = ancestor_dict[labels[j]]
##                print(ancestors)

                indices = [m for m, x in enumerate(labels) if x in ancestors]
##                print(indices)

                for idx in indices:
                    y_pred_scores[a, idx] = y_pred_scores[a, idx] + weights[i] * info_content[0, idx]

    

def complex_KNN(X_train, Y_train, X_test, labels, k, ancestor_dict, info_content):
    """
    Implements GO-aware scoring for each target protein based on KNN 

    Args:
    - X_train: ndarray of shape (n_samples, n_features), training features
    - Y_train: ndarray of shape (n_samples, n_labels), binary label matrix for training data
    - X_test: ndarray of shape (m_samples, n_features), test features
    - k: number of nearest neighbors

    Returns:
    - Y_scores: ndarray of shape (m_samples, n_labels), normalized scores that will be used to form binary prediction matrix
    """

    n_samples, n_labels = Y_train.shape

    # "Fitting" k-NN model on training features and getting cosine distances for weighting
    nn_model = sklearn.neighbors.NearestNeighbors(n_neighbors=k, metric='cosine')
    nn_model.fit(X_train)
    test_distances, test_neighbors = nn_model.kneighbors(X_test, return_distance=True)

    y_pred_scores = np.zeros((X_test.shape[0], n_labels))

    for a in range(X_test.shape[0]): # iterate over datapoints adding votes where neighbors have 1 with GO propagation
        neighbor_labels = Y_train[test_neighbors[a,:]]
    
        weights = (1 / (test_distances[a,:] + 1e-8))     # Getting inverse distance weights from distances. Adding e-6 prevents invalid division
##        print(f"weights: {weights}")
        
        prop_labels(labels, k, neighbor_labels, ancestor_dict, info_content, weights, y_pred_scores, a)

            
        y_pred_scores[a,:] = y_pred_scores[a,:]/(np.sum(y_pred_scores[a,:]+ 1e-8)) # Normalizing each row's scores so no score exceeds 1

    y_pred_scores = pd.DataFrame(y_pred_scores, columns=labels)

    return y_pred_scores

# getting validation scores for different number of nearest neighbors and storing for evaluation with different vote cutoffs

val_y_scoresk1 = complex_KNN(X_train, Y_train, X_val, labels, 1, ancestor_dict, info_content)
val_y_scoresk1.to_csv('IC_val_y_scoresk1_weighted_sum_normalization.csv', index=False)

val_y_scoresk2 = complex_KNN(X_train, Y_train, X_val, labels, 2, ancestor_dict, info_content)
val_y_scoresk2.to_csv('IC_val_y_scoresk2_weighted_sum_normalization.csv', index=False)

val_y_scoresk3 = complex_KNN(X_train, Y_train, X_val, labels, 3, ancestor_dict, info_content)
val_y_scoresk3.to_csv('IC_val_y_scoresk3_weighted_sum_normalization.csv', index=False)

val_y_scoresk5 = complex_KNN(X_train, Y_train, X_val, labels, 5, ancestor_dict, info_content)
val_y_scoresk5.to_csv('IC_val_y_scoresk5_weighted_sum_normalization.csv', index=False)

val_y_scoresk10 = complex_KNN(X_train, Y_train, X_val, labels, 10, ancestor_dict, info_content)
val_y_scoresk10.to_csv('IC_val_y_scoresk10_weighted_sum_normalization.csv', index=False)


