import pandas as pd
import numpy as np
from math import log2


label_matrix = pd.read_csv(r"C:\Users\hb102\Documents\Python\Train\y_train.csv").to_numpy()
print(label_matrix[:10,:10])

labels = pd.read_csv(r"C:\Users\hb102\OneDrive\Desktop\CAFA-5\Train\label_matrix_1500.csv", nrows=0).columns.tolist()[1:]
print(labels)

def calculate_information_content(labels_matrix):
    """
    Calculate the information content (in bits) for each label in a binarized multi-label dataset.
    
    Args:
        labels_matrix: 2D numpy array or pandas DataFrame of shape (n_samples, n_labels)
                      where 1 indicates presence of label and 0 indicates absence
    
    Returns:
        Dictionary with labels as keys and their information content as values
    """
    # Convert to numpy array if it's a DataFrame
    if isinstance(labels_matrix, pd.DataFrame):
        labels_matrix = labels_matrix.values
    
    n_samples = labels_matrix.shape[0]
    n_labels = labels_matrix.shape[1]
    
    # Calculating label frequencies (probability of each label being present)
    label_frequencies = np.sum(labels_matrix, axis=0) / n_samples
    
    # Calculating  information content for each label
    information_content = np.empty((1, labels_matrix.shape[1]))
    for i in range(n_labels):
        p = label_frequencies[i]
        # Handle edge cases where p is 0 or 1 (to avoid log(0))
        if p == 0 or p == 1:
            information_content[0, i] = 0
        else:
            # Information content is -log2(p) for presence and -log2(1-p) for absence
            # We calculate the expected information content
            info_presence = -log2(p)
            info_absence = -log2(1 - p)
            expected_info = p * info_presence + (1 - p) * info_absence
            information_content[0, i] = expected_info
    
    return information_content

information_content = pd.DataFrame(calculate_information_content(label_matrix), columns = labels)

print(information_content)

information_content.to_csv('information_content.csv', index=False)

