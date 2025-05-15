import numpy as np
import pandas as pd
import sklearn
from skmultilearn.adapt import MLkNN
import scipy
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score

X_train = pd.read_csv(r"C:\Users\hb102\Documents\Python\X_train_principal.csv", header = 0).to_numpy()

y_train = pd.read_csv(r"C:\Users\hb102\Documents\Python\y_train.csv", header = 0).to_numpy()

X_val = pd.read_csv(r"C:\Users\hb102\Documents\Python\X_val_principal.csv", header = 0).to_numpy()
y_val = pd.read_csv(r"C:\Users\hb102\Documents\Python\y_val.csv", header = 0).to_numpy()

# MutliLabel-kNN implementation
def custom_mlknn(X_train, Y_train, X_test, k=10, s=1.0):
    """
    Implements MutliLabel-kNN for multi-label classification 

    Parameters:
    - X_train: ndarray of shape (n_samples, n_features), training features
    - Y_train: ndarray of shape (n_samples, n_labels), binary label matrix
    - X_test: ndarray of shape (m_samples, n_features), test features
    - k: number of nearest neighbors
    - s: smoothing parameter for Laplace correction

    Returns:
    - Y_pred: ndarray of shape (m_samples, n_labels), binary predictions
    """

    n_samples, n_labels = Y_train.shape

    # Fit k-NN model on training features
    nn_model = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn_model.fit(X_train)
    neighbors = nn_model.kneighbors(X_train, return_distance=False)  # Indices of k-nearest neighbors

    # Estimate prior probabilities for each label 
    prior_prob_true = (s + Y_train.sum(axis=0)) / (s * 2 + n_samples)  # P(y=1)
    prior_prob_false = 1 - prior_prob_true                            # P(y=0)

    # Estimate conditional probabilities: P(n neighbors have label l | y=1/0)
    max_k = k + 1  # Support up to k neighbors having a label
    count_true = np.zeros((n_labels, max_k))   # Count when y=1
    count_false = np.zeros((n_labels, max_k))  # Count when y=0

    for i in range(n_samples):
        neighbor_labels = Y_train[neighbors[i]]          # Labels of k-nearest neighbors
        label_sum = neighbor_labels.sum(axis=0)          # Count of neighbors having each label

        for l in range(n_labels):
            count = int(label_sum[l])                    
            if Y_train[i, l] == 1:
                count_true[l, count] += 1                
            else:
                count_false[l, count] += 1               

    # Apply Laplace smoothing to conditional probabilities
    cond_prob_true = (s + count_true) / (s * 2 + count_true.sum(axis=1, keepdims=True))     # P(c | y=1)
    cond_prob_false = (s + count_false) / (s * 2 + count_false.sum(axis=1, keepdims=True))  # P(c | y=0)

    # Predict labels for test set
    test_neighbors = nn_model.kneighbors(X_test, return_distance=False)
    Y_pred = np.zeros((X_test.shape[0], n_labels), dtype=int)

    for i in range(X_test.shape[0]):
        neighbor_labels = Y_train[test_neighbors[i]]
        label_sum = neighbor_labels.sum(axis=0).astype(int)  # Count of neighbors with each label

        for l in range(n_labels):
            c = min(label_sum[l], k)  # Ensure index is within range
    # Compute posterior probabilities (Bayes rule, ignoring denominator)
            prob_true = prior_prob_true[l] * cond_prob_true[l, c]
            prob_false = prior_prob_false[l] * cond_prob_false[l, c]
            Y_pred[i, l] = 1 if prob_true > prob_false else 0  # Predict label based on higher posterior prob

    return Y_pred

accuracies = np.empty(10)
f1_scores = np.empty(10)

for i in range(1, 11):
    y_pred = custom_mlknn(X_train, y_train, X_val, k = i)
##    accuracy = accuracy_score(y_val, y_pred)  

    accuracies[i-1] = accuracy_score(y_val, y_pred)
    print(f'Accuracy: {accuracies[i-1]}')
    
    f1_scores[i-1] = f1_score(y_val, y_pred, average='micro')
    print(f"Validation Micro-F1 Score: {f1_scores[i-1]:.4f}")


# Figure for accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), accuracies, marker='o', color='orange', label='Accuracy')
plt.xticks(range(1, 11))  # Set x-ticks to correspond to k values
plt.ylim(0, 1)  
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Score')
plt.title('Model Accuracy vs. Number of Neighbors')
plt.grid()
plt.legend()
plt.show()

# Figure for F1 score
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), f1_scores, marker='o', color='blue', label='F1 Score')
plt.xticks(range(1, 11))  # Set x-ticks to correspond to k values
plt.ylim(0, 1)  # Set y-axis limits
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Score')
plt.title('Micro-F1 Scores vs. Number of Neighbors')
plt.grid()
plt.legend()
plt.show()

# Generate plot
plt.plot(range(1, 21), accuracies, label='Overall Accuracies')  
plt.plot(range(1, 21), f1_scores, label='Validation Micro-F1 Score')  
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.title('KNN accuracy using Equal Weights')
plt.show()

