import matplotlib.pyplot as plt
import csv
import os

def generate_histogram_plot():
    file_name = '../assets/multiclass_dataset_4k.csv'
    
    benign_lengths = []
    attack_lengths = []
    
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
                        label = row[-1].strip().upper()
                        
                        if 'BENIGN' in label:
                            benign_lengths.append(val)
                        else:
                            attack_lengths.append(val)
                    except ValueError:
                        continue

    plt.figure(figsize=(10, 6))
    
    plt.hist(benign_lengths, bins=50, alpha=0.6, label='Normal (BENIGN)', color='#4C72B0')
    plt.hist(attack_lengths, bins=50, alpha=0.6, label='Anomalous (DDoS)', color='#C44E52')
    
    plt.title('Packet Length Distribution', fontsize=18, pad=15)
    plt.xlabel('Packet Length [Bytes]', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    
    plt.legend(loc='upper right', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('../assets/plots/packet_length_histogram.png', dpi=300)

if __name__ == "__main__":
    generate_histogram_plot()