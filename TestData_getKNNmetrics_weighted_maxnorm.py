import sklearn
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Y_test = pd.read_csv(r"C:\Users\hb102\Documents\Python\Train\y_test.csv", header = 0).to_numpy()
label_matrix = pd.read_csv(r"C:\Users\hb102\OneDrive\Desktop\CAFA-5\Train\label_matrix_1500.csv", nrows=0)
label_list = label_matrix.columns.tolist()[1:]

# loading KNN score files we want to evaluate. Note that we only used weighted sum propagation on test data based on validation results
test_y_scores_k1 = pd.read_csv(r"C:\Users\hb102\Documents\Python\test_y_scoresk1_weighted_max_normalization.csv", header = 0).to_numpy()

test_y_scores_k2 = pd.read_csv(r"C:\Users\hb102\Documents\Python\test_y_scoresk2_weighted_max_normalization.csv", header = 0).to_numpy()

test_y_scores_k3 = pd.read_csv(r"C:\Users\hb102\Documents\Python\test_y_scoresk3_weighted_max_normalization.csv", header = 0).to_numpy()

test_y_scores_k5 = pd.read_csv(r"C:\Users\hb102\Documents\Python\test_y_scoresk5_weighted_max_normalization.csv", header = 0).to_numpy()

test_y_scores_k10 = pd.read_csv(r"C:\Users\hb102\Documents\Python\test_y_scoresk10_weighted_max_normalization.csv", header = 0).to_numpy()

files = [test_y_scores_k1, test_y_scores_k2, test_y_scores_k3, test_y_scores_k5, test_y_scores_k10]

def get_scores(Y_pred_scores, cutoff, Y_test):
    """ Function converts score matrix to binary prediction matrix, returns microlocal F1 scores and accuracy of y_pred_scores for a set cutoff and creates graph of progression

    Args:
        filename: .csv filename containing GO-aware KNN scores
        cutoff: cutoff boundary between 0 and 1. Values < cutoff will be converted to 0s, values > cutoff will be converted to 1s
        Y_test: ndarray of actual annotations in multilabel binary form

    Returns:
        Y_pred: multilabel binary matrix associated with input scores
        accuracy: accuracy rate associated with KNN algorithm that produced  at given threshold
        local_F1
    """

    Y_pred = (Y_pred_scores > cutoff).astype(int)

    f1 = sklearn.metrics.f1_score(Y_test, Y_pred, average="samples")
    accuracy = 1- sklearn.metrics.hamming_loss(Y_test, Y_pred) # computing accuracy using 1 - hamming_loss as alternative to accuracy_score
    # accuracy corresponds to fraction of correctly predicted labels over entire dataset
    precision = sklearn.metrics.precision_score(Y_test, Y_pred, average='samples', zero_division = 0)
    f1_weighted = sklearn.metrics.f1_score(Y_test, Y_pred, average='weighted')
    exact_match_rate = np.mean(np.all(Y_pred == Y_test, axis = 1))
    
    return f1, accuracy, precision, f1_weighted, exact_match_rate

weighted_scores = np.empty(5)

weighted_accuracies = np.empty(5)

weighted_precisions = np.empty(5)

weighted_f1_weighteds = np.empty(5)

weighted_exact_rates = np.empty(5)

cutoffs = list(np.arange(0.0000, 0.1200, 0.0200))
##cutoffs = list(np.arange(1e-6, 1e-2, 1e-4)) #for testing much smaller cutoffs as indicated by 0-0.20 results

k_vals = 1,2,3,5,10]

with open("TEST_KNNMetrics_Refined_Range_0.00_to_0.02", "w") as text_file:

    for h in range(len(cutoffs)):

            cutoff = cutoffs[h]

            for i in range(len(k_vals)):
                f1_weighted, accuracy_weighted, precision_weighted, f1_weighted_weighted, exact_rate_weighted = get_scores(files[i], cutoff, Y_test)
                weighted_scores[i] = f1_weighted
                weighted_accuracies[i] = accuracy_weighted
                weighted_precisions[i] = precision_weighted
                weighted_f1_weighteds[i] = f1_weighted_weighted
                weighted_exact_rates[i] = exact_rate_weighted

                    print(f"METRICS FOR K = {k_vals[i]} \n {cutoff:.4f} cutoff F1-score (sample average) = {f1_weighted} \n {cutoff:.4f} cutoff Accuracy = {accuracy_weighted} \n {cutoff:.4f} cutoff Precision (sample average) = {precision_weighted} \n {cutoff:.4f} cutoff Weighted F1-score = {f1_weighted_weighted} \n {cutoff:.4f} cutoff Exact Match Rate = {exact_rate_weighted} \n \n")

                    print(f"METRICS FOR K = {k_vals[i]} \n {cutoff:.4f} cutoff F1-score (sample average) = {f1_weighted} \n {cutoff:.4f} cutoff Accuracy = {accuracy_weighted} \n {cutoff:.4f} cutoff Precision (sample average) = {precision_weighted} \n {cutoff:.4f} cutoff Weighted F1-score = {f1_weighted_weighted} \n {cutoff:.4f} cutoff Exact Match Rate = {exact_rate_weighted} \n \n", file = text_file)


                # F1 Plots    
                fig1, ax1 = plt.subplots(1, 1, figsize = (15, 10))
                ax1.plot((1, 2, 3, 5, 10), weighted_scores, marker='o', color='orange', label='F1-Score')
                ax1.set_xticks((1, 2, 3, 5, 10))  # Set x-ticks to correspond to k values
                ax1.set_ylim(0, 1)  # Set y-axis limits
                ax1.set_xlabel('Number of Neighbors (k)')
                ax1.set_ylabel('Score')
                ax1.set_title(f"Distance Weighted GO-Aware KNN F1-Scores - Test \n Vote Threshold= {cutoff:.4f} \n Max Normalization", 
                        pad=12, 
                        fontsize=12
                )
                ax1.grid(True)
                ax1.legend()

                plt.tight_layout(pad=5.0, h_pad=2.0, w_pad=3.0)
                plt.subplots_adjust(
                        top=0.85,    
                        bottom=0.15, 
                        left=0.07,   
                        right=0.93  
                )
                plt.savefig(f'test_f1_{cutoff:.4f}.png')


        ##        # Accuracy Plots
        ##        fig2, ax2 = plt.subplots(1, 1, figsize = (15, 10))
        ##        ax2.plot((1, 2, 3, 5, 10), weighted_accuracies, marker='o', color='green', label='Accuracy')
        ##        ax2.set_xticks((1, 2, 3, 5, 10))  # Set x-ticks to correspond to k values
        ##        ax2.set_ylim(0, 1)  # Set y-axis limits
        ##        ax2.set_xlabel('Number of Neighbors (k)')
        ##        ax2.set_ylabel('Score')
        ##        ax2.set_title(f"Distance Weighted GO-Aware KNN Accuracy - Test \n Vote Threshold= {cutoff:.4f} \n Max Normalization", 
        ##                pad=12, 
        ##                fontsize=12
        ##        )
        ##        ax2.grid(True)
        ##        ax2.legend()
        ##
        ##        plt.tight_layout(pad=5.0, h_pad=2.0, w_pad=3.0)
        ##        plt.subplots_adjust(
        ##                top=0.85,    
        ##                bottom=0.15, 
        ##                left=0.07,   
        ##                right=0.93  
        ##        )
        ##        plt.savefig(f'test_acc_{cutoff:.4f}.png')


                # Precision Plots
                fig3, ax3 = plt.subplots(1, 1, figsize = (15, 10))
                ax3.plot((1, 2, 3, 5, 10), weighted_precisions, marker='o', color='green', label='Precision')
                ax3.set_xticks((1, 2, 3, 5, 10))  # Set x-ticks to correspond to k values
                ax3.set_ylim(0, 1)  # Set y-axis limits
                ax3.set_xlabel('Number of Neighbors (k)')
                ax3.set_ylabel('Score')
                ax3.set_title(f"Distance Weighted GO-Aware KNN Precision - Test \n Vote Threshold= {cutoff:.4f} \n Max Normalization", 
                        pad=12, 
                        fontsize=12
                )
                ax3.grid(True)
                ax3.legend()


                plt.tight_layout(pad=5.0, h_pad=2.0, w_pad=3.0)
                plt.subplots_adjust(
                        top=0.85,    
                        bottom=0.15, 
                        left=0.07,   
                        right=0.93  
                )
                plt.savefig(f'test_precision_{cutoff:.4f}.png')


                # Weighted F1 Plots
                fig4, ax4 = plt.subplots(1, 1, figsize = (15, 10))
                ax4.plot((1, 2, 3, 5, 10), weighted_f1_weighteds, marker='o', color='green', label='Weighted F1')
                ax4.set_xticks((1, 2, 3, 5, 10))  # Set x-ticks to correspond to k values
                ax4.set_ylim(0, 1)  # Set y-axis limits
                ax4.set_xlabel('Number of Neighbors (k)')
                ax4.set_ylabel('Score')
                ax4.set_title(f"Distance Weighted GO-Aware KNN Weighted F1-Score - Test \n Vote Threshold= {cutoff:.4f} \n Max Normalization", 
                        pad=12, 
                        fontsize=12
                )
                ax4.grid(True)
                ax4.legend()

                plt.tight_layout(pad=5.0, h_pad=2.0, w_pad=3.0)
                plt.subplots_adjust(
                        top=0.85,    
                        bottom=0.15, 
                        left=0.07,   
                        right=0.93  
                )
                plt.savefig(f'test_weightedf1_{cutoff:.4f}.png')


                # Exact Match Rate Plots
                fig5, ax5 = plt.subplots(1, 1, figsize = (15, 10))
                ax5.plot((1, 2, 3, 5, 10), weighted_exact_rates, marker='o', color='green', label='Exact Match Rate')
                ax5.set_xticks((1, 2, 3, 5, 10))  # Set x-ticks to correspond to k values
                ax5.set_ylim(0, 1)  # Set y-axis limits
                ax5.set_xlabel('Number of Neighbors (k)')
                ax5.set_ylabel('Rate')
                ax5.set_title(f"Distance Weighted GO-Aware KNN Match Rate - Test \n Vote Threshold= {cutoff:.4f} \n Max Normalization", 
                    pad=12, 
                    fontsize=12
                )
                ax5.grid(True)
                ax5.legend()
                
                plt.tight_layout(pad=5.0, h_pad=2.0, w_pad=3.0)
                plt.subplots_adjust(
                        top=0.85,    
                        bottom=0.15, 
                        left=0.07,   
                        right=0.93  
                )
                plt.savefig(f'test_exactmatch_{cutoff:.4f}.png')
