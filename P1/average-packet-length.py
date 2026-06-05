import matplotlib.pyplot as plt
import csv
import os

def generate_average_length_plot():
    file_name = '../multiclass_dataset_4k.csv'
    
    class_sums = {}
    class_counts = {}
    
    if os.path.exists(file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            
            target_column = -1
            current_col = 0
            
            for col in header:
                if 'Fwd Packet Length Max' in col:
                    target_column = current_col
                    break
                current_col += 1
                
            if target_column != -1:
                for row in reader:
                    if not row:
                        continue
                        
                    try:
                        val = float(row[target_column])
                        label = row[-1].strip()
                        
                        if label in class_sums:
                            class_sums[label] += val
                            class_counts[label] += 1
                        else:
                            class_sums[label] = val
                            class_counts[label] = 1
                    except ValueError:
                        continue
                        
    labels = []
    for key in class_sums.keys():
        labels.append(key)
        
    averages = []
    for key in labels:
        avg = class_sums[key] / class_counts[key]
        averages.append(avg)
        
    color_map = {}
    color_map['BENIGN'] = '#2ca02c'
    color_map['DRDOS_DNS'] = '#1f77b4'
    color_map['SYN'] = '#ff7f0e'
    color_map['UDP-LAG'] = '#9467bd'
    color_map['TFTP'] = '#8c564b'
    
    colors = []
    for label in labels:
        matched = False
        for key, color in color_map.items():
            if key in label.upper():
                colors.append(color)
                matched = True
                break
        
        if not matched:
            colors.append('#333333')
            
    plt.figure(figsize=(12, 7))
    
    bars = plt.bar(labels, averages, color=colors, alpha=0.85)
    
    plt.title('Average Packet Length per Traffic Class', fontsize=18, pad=20)
    plt.xlabel('Traffic Class', fontsize=14)
    plt.ylabel('Average Length [Bytes]', fontsize=14)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    max_avg = 0
    for avg in averages:
        if avg > max_avg:
            max_avg = avg
            
    plt.ylim(0, max_avg * 1.15)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max_avg * 0.02), f"{yval:.1f}", ha='center', va='bottom', fontsize=13, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('average_packet_length.png', dpi=300)

if __name__ == "__main__":
    generate_average_length_plot()