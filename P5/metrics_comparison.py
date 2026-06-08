import matplotlib.pyplot as plt
import numpy as np
import csv

def plot_all_metrics_comparison():
    input_file = '../assets/results/combined_metrics_results.csv'
    
    models = []
    models.append('Gaussian Naive Bayes')
    models.append('K-Nearest Neighbors')
    models.append('Decision Tree')
    
    metrics = []
    metrics.append('precision_mean')
    metrics.append('recall_mean')
    metrics.append('f1_score_mean')
    metrics.append('balanced_accuracy_mean')
    
    metric_titles = []
    metric_titles.append('Precision')
    metric_titles.append('Recall')
    metric_titles.append('F1-Score')
    metric_titles.append('Balanced Accuracy')
    
    data_clean = {}
    data_leak = {}
    
    for metric in metrics:
        data_clean[metric] = {}
        data_leak[metric] = {}
        for model in models:
            data_clean[metric][model] = 0.0
            data_leak[metric][model] = 0.0
            
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        col_indices = {}
        current_idx = 0
        for col in header:
            col_indices[col] = current_idx
            current_idx += 1
            
        for row in reader:
            if not row:
                continue
                
            exp_type = row[col_indices['experiment_type']]
            model_name = row[col_indices['classifier']]
            
            for metric in metrics:
                val = float(row[col_indices[metric]])
                if exp_type == 'CLEAN_DATA':
                    data_clean[metric][model_name] = val
                else:
                    data_leak[metric][model_name] = val
                    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat = axes.flatten()
    
    x_positions = np.arange(len(models))
    bar_width = 0.35
    
    for index in range(len(metrics)):
        current_metric = metrics[index]
        current_title = metric_titles[index]
        ax = axes_flat[index]
        
        clean_values = []
        leak_values = []
        
        for model in models:
            clean_values.append(data_clean[current_metric][model])
            leak_values.append(data_leak[current_metric][model])
            
        bars_clean = ax.bar(x_positions - bar_width/2, clean_values, bar_width, label='Clean Data', color='#1f77b4')
        bars_leak = ax.bar(x_positions + bar_width/2, leak_values, bar_width, label='Data Leak', color='#d62728')
        
        ax.set_title(current_title, fontsize=16, pad=15)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(models, fontsize=12)
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        if index == 0:
            ax.legend(fontsize=12, loc='upper left')
            
        for bar in bars_clean:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom', fontsize=11, fontweight='bold', color='#1f77b4')
            
        for bar in bars_leak:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom', fontsize=11, fontweight='bold', color='#d62728')
            
    fig.suptitle('Metrics Comparison (Clean vs Data Leak)', fontsize=20, y=0.98)
    plt.tight_layout(pad=2.0)
    plt.savefig('../assets/plots/metrics_comparison.png', dpi=300)

if __name__ == "__main__":
    plot_all_metrics_comparison()