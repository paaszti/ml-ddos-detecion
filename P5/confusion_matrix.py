import matplotlib.pyplot as plt
import numpy as np
import csv

def plot_all_confusion_matrices():
    input_file = '../assets/results/combined_confusion_matrices.csv'
    
    experiments = []
    experiments.append('CLEAN_DATA')
    experiments.append('DATA_LEAK')
    
    classifiers = []
    classifiers.append('Gaussian Naive Bayes')
    classifiers.append('K-Nearest Neighbors')
    classifiers.append('Decision Tree')
    
    data_storage = {}
    unique_labels = set()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        for row in reader:
            if not row:
                continue
            exp = row[0]
            clf = row[1]
            true_lbl = row[2]
            pred_lbl = row[3]
            count_val = int(row[4])
            
            unique_labels.add(true_lbl)
            unique_labels.add(pred_lbl)
            
            key = (exp, clf, true_lbl, pred_lbl)
            data_storage[key] = count_val
            
    labels_list = sorted(list(unique_labels))
    num_labels = len(labels_list)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    for exp_idx in range(len(experiments)):
        current_exp = experiments[exp_idx]
        for clf_idx in range(len(classifiers)):
            current_clf = classifiers[clf_idx]
            
            ax = axes[exp_idx, clf_idx]
            matrix = np.zeros((num_labels, num_labels), dtype=int)
            
            for i in range(num_labels):
                t_label = labels_list[i]
                for j in range(num_labels):
                    p_label = labels_list[j]
                    search_key = (current_exp, current_clf, t_label, p_label)
                    matrix[i, j] = data_storage.get(search_key, 0)
                    
            im = ax.imshow(matrix, cmap='Blues', interpolation='nearest')
            
            ax.set_xticks(np.arange(num_labels))
            ax.set_yticks(np.arange(num_labels))
            ax.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=10)
            ax.set_yticklabels(labels_list, fontsize=10)
            
            title_text = current_clf + " (" + current_exp + ")"
            ax.set_title(title_text, fontsize=14, pad=12, fontweight='bold')
            ax.set_xlabel('Predicted Class', fontsize=11)
            ax.set_ylabel('True Class', fontsize=11)
            
            max_val = matrix.max()
            threshold = max_val / 2.0
            
            for i in range(num_labels):
                for j in range(num_labels):
                    val = matrix[i, j]
                    if val > threshold:
                        text_color = 'white'
                    else:
                        text_color = 'black'
                    ax.text(j, i, str(val), ha='center', va='center', color=text_color, fontsize=11, fontweight='bold')
                    
    plt.tight_layout(pad=3.0)
    plt.savefig('../assets/plots/confusion_matrices.png', dpi=300)

if __name__ == "__main__":
    plot_all_confusion_matrices()