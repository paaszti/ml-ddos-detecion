import matplotlib.pyplot as plt
import csv
import os

def generate_cumulative_imbalance_plot():
    files_to_attacks = {
        '../01-12/DrDoS_DNS.csv': 'DrDoS_DNS',
        '../01-12/Syn.csv': 'Syn',
        '../01-12/UDPLag.csv': 'UDP-lag',
        '../01-12/TFTP.csv': 'TFTP'
    }

    total_benign = 0
    attack_counts = {}

    for attack in files_to_attacks.values():
        attack_counts[attack] = 0

    for file_name, attack_name in files_to_attacks.items():
        if not os.path.exists(file_name):
            continue
            
        with open(file_name, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            
            for row in reader:
                if not row:
                    continue
                    
                label = row[-1].strip().upper()
                
                if 'BENIGN' in label:
                    total_benign += 1
                elif attack_name.upper() in label:
                    attack_counts[attack_name] += 1

    labels = ['Total BENIGN'] + list(attack_counts.keys())
    counts = [total_benign] + list(attack_counts.values())
    colors = ['#2ca02c'] + ['#d62728'] * len(attack_counts)

    plt.figure(figsize=(10, 10))
    bars = plt.bar(labels, counts, color=colors)

    plt.yscale('log')
    plt.title('Benign Traffic vs DDoS Attacks', fontsize=14)
    plt.ylabel('Number of Packets (Log Scale)', fontsize=12)
    plt.xlabel('Traffic Type', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval * 1.2, f"{int(yval):,}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('cumulative_imbalance.png', dpi=300)

if __name__ == "__main__":
    generate_cumulative_imbalance_plot()