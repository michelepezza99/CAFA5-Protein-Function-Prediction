import numpy as np
import pandas as pd
import sklearn
from skmultilearn.adapt import MLkNN
from sklearn.neighbors import KNeighborsClassifier
import json

X_train = pd.read_csv(r"C:\Users\hb102\Documents\Python\Train\X_train_principal.csv", header = 0).to_numpy()

Y_train = pd.read_csv(r"C:\Users\hb102\Documents\Python\Train\y_train.csv", header = 0).to_numpy()

X_val = pd.read_csv(r"C:\Users\hb102\Documents\Python\Train\X_val_principal.csv", header = 0).to_numpy()

label_matrix = pd.read_csv(r"C:\Users\hb102\OneDrive\Desktop\CAFA-5\Train\label_matrix_1500.csv", nrows=0)
labels = label_matrix.columns.tolist()[1:]  

def simple_KNN(X_train, Y_train, X_test, labels, k):
    """
    Implements basic weighted scoring for each target protein based on KNN, not GO-aware

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
        
        for i in range(k):
            single_label = neighbor_labels[i]
        
            for j in range(neighbor_labels.shape[1]):
                if single_label[j] == 1:
                    y_pred_scores[a, j] = y_pred_scores[a, j] + weights[i]
        
##        if a == 3: # testing output from prop_labels
##            print(y_pred_scores[a,:])
            
        y_pred_scores[a,:] = y_pred_scores[a,:]/(max(y_pred_scores[a,:]+ 1e-8)) # Normalizing each row's scores so no score exceeds 1

    y_pred_scores = pd.DataFrame(y_pred_scores, columns=labels)

    return y_pred_scores

# getting validation scores for different number of nearest neighbors and storing for evaluation with different vote cutoffs

no_prop_val_y_scoresk1 = simple_KNN(X_train, Y_train, X_val, labels, 1)
no_prop_val_y_scoresk1.to_csv('no_prop_val_y_scoresk1.csv', index=False)

no_prop_val_y_scoresk2 = simple_KNN(X_train, Y_train, X_val, labels, 2)
no_prop_val_y_scoresk2.to_csv('no_prop_val_y_scoresk2.csv', index=False)

no_prop_val_y_scoresk3 = simple_KNN(X_train, Y_train, X_val, labels, 3)
no_prop_val_y_scoresk3.to_csv('no_prop_val_y_scoresk3.csv', index=False)

no_prop_val_y_scoresk5 = simple_KNN(X_train, Y_train, X_val, labels, 5)
no_prop_val_y_scoresk5.to_csv('no_prop_val_y_scoresk5.csv', index=False)

no_prop_val_y_scoresk10 = simple_KNN(X_train, Y_train, X_val, labels, 10)
no_prop_val_y_scoresk10.to_csv('no_prop_val_y_scoresk10.csv', index=False)
