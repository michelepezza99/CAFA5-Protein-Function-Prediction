import numpy as np
import pandas as pd
import sklearn
from skmultilearn.adapt import MLkNN
from sklearn.decomposition import PCA
import scipy
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

"""
Prior to experimenting with different KNN models, we used principal component analysis to reduced the dimensionality of our data.
Using the training data as input, PCA determined that 113 features were necessary to explain 95% of the variance in the data.
We then tranformed all three datasets based on this information.

Citation for Multilearn library: @article{zhang2007ml,
  title={ML-KNN: A lazy learning approach to multi-label learning},
  author={Zhang, Min-Ling and Zhou, Zhi-Hua},
  journal={Pattern recognition},
  volume={40},
  number={7},
  pages={2038--2048},
  year={2007},
  publisher={Elsevier}
}
"""


X_train = pd.read_csv(r"C:\Users\hb102\Documents\Python\X_train_principal.csv", header = 0).to_numpy()
y_train = pd.read_csv(r"C:\Users\hb102\Documents\Python\y_train.csv", header = 0).to_numpy()

X_val = pd.read_csv(r"C:\Users\hb102\Documents\Python\X_val_principal.csv", header = 0).to_numpy()
y_val = pd.read_csv(r"C:\Users\hb102\Documents\Python\y_val.csv", header = 0).to_numpy()

X_test = pd.read_csv(r"C:\Users\hb102\Documents\Python\X_test_principal.csv", header = 0).to_numpy()
y_test = pd.read_csv(r"C:\Users\hb102\Documents\Python\y_test.csv", header = 0).to_numpy()



def inverse_sqrt(distances):
    return 1/np.sqrt(distances)


"""This is my attempt to write a dynamic knn function so far where each point can be predicted by no more than 10 neighbors. Based on
the method shown here : https://rabmcmenemy.medium.com/unveiling-dynamic-weighted-knn-a-deep-exploration-of-adaptive-and-weighted-k-nearest-neighbours-3957ead074a8""" 
"""Still not functional but will continue to work on this."""

## def dynamic_knn(X_train, y_train, X_val, y_val, k_max=10):  
##    npoints = np.shape(X_train)[0]
##    distances = np.empty((npoints, npoints))
##
##    for i in range(npoints):  
##        for j in range(npoints):
##            distances[i,j] = np.linalg.norm(X_train[i,:]-X_train[j,:])  
##
##    if not scipy.linalg.issymmetric(distances):
##        print("faulty distance matrix")
##                                                                                        
##    sorted_distances = np.sort(distances)[:,0:k_max]
##
##    density_thresholds = np.sum(sorted_distances, axis=-1)/k_max
##
##    adaptive_k = np.empty((npoints,1))  
##
##    dynamic_train_accuracy = np.empty(npoints) 
##    dynamic_val_accuracy = np.empty(npoints)
##
##    for i in range(npoints):
##        adaptive_k[i] = np.sum(distances[i,:] <= density_thresholds[i])
##        
##        dynamic_knn = KNeighborsClassifier(n_neighbors=int(adaptive_k[i]))  
##        dynamic_knn.fit(X_train, y_train)
##
##        dynamic_train_accuracy[i] = dynamic_knn.score(X_train, y_train)
##        dynamic_val_accuracy[i] = dynamic_knn.score(X_val, y_val)
##
##    return dynamic_train_accuracy, dynamic_val_accuracy


""" Testing performance of different k-vals with equal weights"""
equald_train_accuracy = np.empty(10)
equald_val_accuracy = np.empty(10)

for i in range(1, 11):  
    equald_knn = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
    
    equald_train_accuracy[i-1] = equald_knn.score(X_train, y_train)
    equald_val_accuracy[i-1] = equald_knn.score(X_val, y_val)

# Generate plot
plt.plot(range(10), equald_val_accuracy, label='Validation accuracy')  
plt.plot(range(10), equald_train_accuracy, label='Training accuracy')  
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.title('KNN accuracy using Equal Weights')
plt.show()




""" Testing performance of different k-vals with inverse Euclidean distance weights"""
inversed_train_accuracy = np.empty(10)
inversed_val_accuracy = np.empty(10)

for i in range(1, 11):
    inversed_knn = KNeighborsClassifier(n_neighbors=i, weights='distance').fit(X_train, y_train)
    
    inversed_train_accuracy[i-1] = inversed_knn.score(X_train, y_train)
    inversed_val_accuracy[i-1] = inversed_knn.score(X_val, y_val)

# Generate plot
plt.plot(range(10), inversed_val_accuracy, label='Validation accuracy')
plt.plot(range(10), inversed_train_accuracy, label='Training accuracy')  # Fixed variable name
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.title('KNN accuracy using Inverse Distance Weights')
plt.show()


""" Testing performance of different k-vals with square root inverse Euclidean distance weights """
sqrt_inversed_train_accuracy = np.empty(10)
sqrt_inversed_val_accuracy = np.empty(10)

for i in range(1, 11):
    sqrt_inversed_knn = KNeighborsClassifier(n_neighbors=i, weights=inverse_sqrt).fit(X_train, y_train)
    sqrt_inversed_train_accuracy[i-1] = sqrt_inversed_knn.score(X_train, y_train)
    sqrt_inversed_val_accuracy[i-1] = sqrt_inversed_knn.score(X_val, y_val)

# Generate plot
plt.plot(range(10), sqrt_inversed_val_accuracy, label='Validation accuracy')
plt.plot(range(10), sqrt_inversed_train_accuracy, label='Training accuracy')  # Fixed variable name
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.title('KNN accuracy using Inverse Square Root Distance Weights')
plt.show()


"""Comparing Multilabel KNN to Single Label KNeighborsClassifier - This library has different documentation. KNeighbors Classifiers
automatically handles multiple labels but this is a different multilabel library. My syntax is wrong (I think .fit only takes one
positional argument), I just haven't had time to update and figure out where y_train comes in."""
import sklearn.metrics as metrics

mlKNN_train_accuracy = np.empty(10)
mlKNN_val_accuracy = np.empty(10)

for i in range(1, 11):
    mlKNN = MLkNN(k=i)
    mlKNN_model = mlKNN.fit(X_train, y_train)  
    mlKNN_train_accuracy[i-1] = metrics.accuracy_score(y_train, mlKNN_model.predict(X_train))
    mlKNN_val_accuracy[i-1] = metrics.accuracy_score(y_val, mlKNN_model.predict(X_val))

# Generate plot
plt.plot(range(10), mlKNN_val_accuracy, label='Validation accuracy')
plt.plot(range(10), mlKNN_train_accuracy, label='Training accuracy')  
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.title('Multilearn KNN Accuracy')
plt.show()
