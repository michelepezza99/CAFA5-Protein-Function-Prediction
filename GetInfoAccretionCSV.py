import pandas as pd

# Load the label matrix header
label_matrix = pd.read_csv(r"C:\Users\hb102\OneDrive\Desktop\CAFA-5\Train\label_matrix_1500.csv", nrows=0)
ia_data = pd.read_csv(r"C:\Users\hb102\OneDrive\Desktop\CAFA-5\Train\IA.csv")

# Extract column labels without 'protein id' col 
label_columns = label_matrix.columns.tolist()[1:]  

# print(label_columns[:10])  # Print first 10 column names to verify


# Filter ia_data to only include rows where the first column value is in label_columns
# Assuming the first column of ia_data contains the gene ontology labels
ia_labels = ia_data.iloc[:, 0].tolist()

common_terms = list(set(label_columns) & set(ia_labels))
print(f"Found {len(common_terms)} matching GO terms.")

information_accretion = ia_data[ia_data.iloc[:, 0].isin(label_columns)].copy()

information_accretion  = information_accretion.set_index(information_accretion.columns[0]).T

information_accretion = information_accretion.reindex(columns=label_columns) 


# Save the information_accretion DataFrame to a CSV file
information_accretion.to_csv('information_accretion.csv', index=False)

print("information_accretion DataFrame has been saved to 'information_accretion.csv'.")
