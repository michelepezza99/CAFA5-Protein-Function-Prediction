### ✅ Install all required libraries
##!pip install biopython
##!pip install goatools
##!pip install scikit-learn
##!pip install scikit-multilearn
##!pip install pandas
##!pip install numpy
##!pip install matplotlib  # Optional, for any future plotting
##
### ✅ Now you can safely import everything used in your notebook

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Optional if needed

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA  # If used later
from sklearn.metrics import f1_score, precision_score  # For evaluation

from skmultilearn.model_selection import iterative_train_test_split

from Bio import SeqIO
from goatools.obo_parser import GODag

import os  # For managing paths if needed

embeddings = pd.read_csv(r'C:\Users\hb102\Downloads\embeddings.csv')

# Load train_sequences.fasta
from Bio import SeqIO
train_sequences = {}
for record in SeqIO.parse(r'C:\Users\hb102\OneDrive\Desktop\CAFA-5\Train\train_sequences.fasta', 'fasta'):
    train_sequences[record.id] = str(record.seq)
# Load train_terms.tsv
train_terms = pd.read_csv(r'C:\Users\hb102\OneDrive\Desktop\CAFA-5\Train\train_terms.tsv', sep='\t')
# Load go-basic.obo
from goatools.obo_parser import GODag
go_dag = GODag(r'C:\Users\hb102\OneDrive\Desktop\CAFA-5\Train\go-basic.obo')

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

k = 1500  # Number of top GO terms to keep to prevent issues due to magnitude

protein_ids = embeddings["Protein Id"].tolist()

# === Step 2: Load GO Annotations ===
terms_df = pd.read_csv(r'C:\Users\hb102\OneDrive\Desktop\CAFA-5\Train\train_terms.tsv', sep='\t')

# === Step 3: Filter to proteins that have embeddings ===
terms_df = terms_df[terms_df['EntryID'].isin(protein_ids)]

# === Step 4: Keep only the top 1500 GO terms based on frequency ===
top_terms = terms_df['term'].value_counts().nlargest(k).index
terms_df = terms_df[terms_df['term'].isin(top_terms)]

# === Step 5: Group GO terms by protein ===
grouped_terms = terms_df.groupby("EntryID")["term"].apply(list)

# === Step 6: Fill missing proteins with empty lists ===
go_terms_list = [grouped_terms.get(pid, []) for pid in protein_ids]

# === Step 7: Create binary matrix ===
mlb = MultiLabelBinarizer(classes=sorted(top_terms))
Y = mlb.fit_transform(go_terms_list)
go_terms = mlb.classes_

# === Step 8: Turn into a DataFrame ===
label_df = pd.DataFrame(Y, columns=go_terms)
label_df.insert(0, "Protein Id", protein_ids)

# === (Optional) Save for reuse ===
label_df.to_csv("label_matrix_1500.csv", index=False)

print(f"✅ Label matrix shape: {label_df.shape}")

def get_go_terms(protein_id):
    """
    Given a protein ID, return the corresponding GO terms.
    """
    if protein_id in label_df["Protein Id"].values:
        # Get the row corresponding to the protein ID
        row = label_df[label_df["Protein Id"] == protein_id].iloc[:, 1:]
        # Extract the GO terms where the value is 1
        go_terms = row.columns[row.values.flatten() == 1].tolist()
        return go_terms
    else:
        return None

# Example usage
protein_id = "P20536"  # Replace with a valid protein ID from your dataset
go_terms = get_go_terms(protein_id)
print(f"GO terms for protein {protein_id}: {go_terms}")

from sklearn.model_selection import train_test_split
import numpy as np

# Step 1: Prepare X, Y
X = embeddings.drop("Protein Id", axis=1).values
Y = label_df.drop("Protein Id", axis=1).values
protein_ids = np.array(embeddings["Protein Id"])

# Step 2: Simple random split (no stratify)
X_remain, X_test, y_remain, y_test, id_remain, id_test = train_test_split(
    X, Y, protein_ids, test_size=0.10, random_state=42
)

# Step 3: Split train/val
X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
    X_remain, y_remain, id_remain, test_size=1/9, random_state=42
)

# Step 4: Print shapes
print("✅ Final Split:")
print(f"Train:      {X_train.shape}, {y_train.shape}")
print(f"Validation: {X_val.shape}, {y_val.shape}")
print(f"Test:       {X_test.shape}, {y_test.shape}")

# Save embeddings
pd.DataFrame(X_train).to_csv("X_train.csv", index=False)
pd.DataFrame(X_val).to_csv("X_val.csv", index=False)
pd.DataFrame(X_test).to_csv("X_test.csv", index=False)

# Save labels
pd.DataFrame(y_train, columns=label_df.columns[1:]).to_csv("y_train.csv", index=False)
pd.DataFrame(y_val, columns=label_df.columns[1:]).to_csv("y_val.csv", index=False)
pd.DataFrame(y_test, columns=label_df.columns[1:]).to_csv("y_test.csv", index=False)

# Optionally save protein IDs (if needed for debugging)
train_ids = np.array(protein_ids)[np.isin(X, X_train).all(axis=1)]
val_ids = np.array(protein_ids)[np.isin(X, X_val).all(axis=1)]
test_ids = np.array(protein_ids)[np.isin(X, X_test).all(axis=1)]

pd.DataFrame({"Protein Id": train_ids}).to_csv("train_ids.csv", index=False)
pd.DataFrame({"Protein Id": val_ids}).to_csv("val_ids.csv", index=False)
pd.DataFrame({"Protein Id": test_ids}).to_csv("test_ids.csv", index=False)

print(len(X_train))
print(len(y_train))
print(len(X_val))
print(len(X_test))
print(len(y_test))
