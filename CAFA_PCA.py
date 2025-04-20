import numpy as np
import pandas as pd
import sklearn
from skmultilearn.adapt import MLkNN
from sklearn.decomposition import PCA
import scipy
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

X_train = pd.read_csv(r"C:\Users\hb102\Documents\Python\X_train.csv", header = 0)

"Performing PCA on training data"
pca = PCA()
pca.fit(X_train)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

"Find number of components for 95% variance"

n_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components}")


X_val = pd.read_csv(r"C:\Users\hb102\Documents\Python\X_val.csv", header = 0)
X_test = pd.read_csv(r"C:\Users\hb102\Documents\Python\X_test.csv", header = 0)

"Apply PCA with enough components to explain 95% variance"

pca = PCA(n_components)

# Transform both training and test data
X_train_pca = pca.fit_transform(X_train)  # Fit and transform training
X_val_pca = pca.transform(X_val)        # Only transform test data
X_test_pca = pca.transform(X_test)  

print(f"Original training shape: {X_train.shape}")
print(f"PCA-reduced training shape: {X_train_pca.shape}")

print(f"Original val shape: {X_val.shape}")
print(f"PCA-reduced val shape: {X_val_pca.shape}")

print(f"Original test shape: {X_test.shape}")
print(f"PCA-reduced test shape: {X_test_pca.shape}")

X_train_principal = pd.DataFrame(X_train_pca, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
X_val_principal = pd.DataFrame(X_val_pca, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
X_test_principal = pd.DataFrame(X_test_pca, columns=[f"PC{i+1}" for i in range(pca.n_components_)])

X_train_principal.to_csv('X_train_principal.csv', index=False)
X_val_principal.to_csv('X_val_principal.csv', index=False)
X_test_principal.to_csv('X_test_principal.csv', index=False)
