import matplotlib.pyplot as plt
import csv
import os

def generate_balanced_plot():
    file_name = '../multiclass_dataset_4k.csv'
    
    class_counts = {}
    
    if os.path.exists(file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            
            for row in reader:
                if not row:
                    continue
                
                label = row[-1].strip()
                
                if label in class_counts:
                    class_counts[label] += 1
                else:
                    class_counts[label] = 1
                    
    labels = []
    for key in class_counts.keys():
        labels.append(key)
        
    counts = []
    for value in class_counts.values():
        counts.append(value)
        
    color_map = {
        'BENIGN': '#2ca02c',
        'DrDoS_DNS': '#1f77b4',
        'Syn': '#ff7f0e',
        'UDP-lag': '#9467bd',
        'TFTP': '#8c564b'
    }
    
    colors = []
    for label in labels:
        matched = False
        for key, color in color_map.items():
            if key.lower() in label.lower():
                colors.append(color)
                matched = True
                break
        
        if not matched:
            colors.append('#333333')
            
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, counts, color=colors)
    
    plt.title('Balanced Dataset', fontsize=14)
    plt.ylabel('Number of Packets', fontsize=12)
    plt.xlabel('Traffic Type', fontsize=12)
    
    max_count = 0
    for count in counts:
        if count > max_count:
            max_count = count
            
    plt.ylim(0, max_count * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        offset = max_count * 0.05
        plt.text(bar.get_x() + bar.get_width()/2, yval + offset, f"{int(yval):,}", ha='center', va='bottom', fontsize=10)
        
    plt.tight_layout()
    plt.savefig('balanced_dataset.png', dpi=300)

if __name__ == "__main__":
    generate_balanced_plot()