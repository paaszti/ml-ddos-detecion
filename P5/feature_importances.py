import matplotlib.pyplot as plt
import csv

def plot_importance_comparison():
    input_file = '../assets/results/combined_feature_importances.csv'
    
    clean_pairs = []
    leak_pairs = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        
        for row in reader:
            if not row:
                continue
                
            exp_type = row[0]
            feat_name = row[1]
            weight = float(row[2])
            
            pair = (weight, feat_name)
            
            if exp_type == "CLEAN_DATA":
                clean_pairs.append(pair)
            else:
                leak_pairs.append(pair)
                
    clean_pairs.sort(key=lambda item: item[0], reverse=True)
    leak_pairs.sort(key=lambda item: item[0], reverse=True)
    
    top_10_clean_weights = []
    top_10_clean_names = []
    
    for index in range(10):
        if index < len(clean_pairs):
            top_10_clean_weights.append(clean_pairs[index][0])
            top_10_clean_names.append(clean_pairs[index][1])
            
    top_10_leak_weights = []
    top_10_leak_names = []
    
    for index in range(10):
        if index < len(leak_pairs):
            top_10_leak_weights.append(leak_pairs[index][0])
            top_10_leak_names.append(leak_pairs[index][1])
            
    top_10_clean_weights.reverse()
    top_10_clean_names.reverse()
    
    top_10_leak_weights.reverse()
    top_10_leak_names.reverse()
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    ax1 = axes[0]
    bars1 = ax1.barh(top_10_clean_names, top_10_clean_weights, color='#1f77b4')
    ax1.set_title('Clean Data', fontsize=16, pad=15)
    ax1.set_xlabel('Feature Weight (0,0 - 1,0)', fontsize=12)
    ax1.set_ylabel('Feature Name', fontsize=12)
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    
    for bar in bars1:
        width = bar.get_width()
        ax1.text(width + 0.005, bar.get_y() + bar.get_height()/2, f"{width:.4f}", va='center', fontsize=11)
        
    ax2 = axes[1]
    bars2 = ax2.barh(top_10_leak_names, top_10_leak_weights, color='#d62728')
    ax2.set_title('Data Leak', fontsize=16, pad=15)
    ax2.set_xlabel('Feature Weight (0,0 - 1,0)', fontsize=12)
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    
    for bar in bars2:
        width = bar.get_width()
        ax2.text(width + 0.005, bar.get_y() + bar.get_height()/2, f"{width:.4f}", va='center', fontsize=11)
        
    plt.tight_layout()
    plt.savefig('../assets/plots/leak_comparison.png', dpi=300)

if __name__ == "__main__":
    plot_importance_comparison()