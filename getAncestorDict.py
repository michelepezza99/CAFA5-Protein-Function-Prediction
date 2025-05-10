import pandas as pd
import networkx
import obonet
import json
import goatools
from goatools import obo_parser
from goatools.obo_parser import GODag

go = obo_parser.GODag(r"C:\Users\hb102\OneDrive\Desktop\CAFA-5\Train\go-basic.obo")

label_matrix = pd.read_csv(r"C:\Users\hb102\OneDrive\Desktop\CAFA-5\Train\label_matrix_1500.csv", nrows=0)
labels = label_matrix.columns.tolist()[1:] 

def build_ancestor_dict(go_terms = labels, go_graph = go):
    """Build dictionary of ancestors for each GO term in label matrix"""
    ancestor_dict = {}
    
    for term_id in go_terms:
            if term_id in go:
                term = go[term_id]
                # Get all ancestors (convert set to list and include the term itself for scoring)
                ancestors = list(term.get_all_parents())
                ancestors.append(term_id)
                ancestor_dict[term_id] = ancestors
            else:
                print(f"Warning: GO term {term_id} not found in the ontology")
            
            ancestor_dict[term_id] = ancestors

    return ancestor_dict

ancestor_dict = build_ancestor_dict(labels, go)

with open("ancestor_dict.json", 'w') as f:
    json.dump(ancestor_dict, f)
