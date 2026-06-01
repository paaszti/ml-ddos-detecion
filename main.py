import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt


raw_data = np.genfromtxt('small_DrDoS_DNS.csv', delimiter=',', dtype=str, skip_header=1)

y_text = raw_data[:, -1]

X_text = raw_data[:, :-1]

with open('small_DrDoS_DNS.csv', 'r') as f:
    header = f.readline().strip().split(',')

banned_columns = [
    'Unnamed: 0',       
    'Flow ID',          
    'Source IP',        
    'Source Port',      
    'Destination IP',   
    'Destination Port', 
    'Timestamp',        
    'SimillarHTTP',     
    'Inbound'           
]

del_index = []

for i, name in enumerate(header[:-1]): 
    clean_name = name.strip() 
    if clean_name in banned_columns:
        del_index.append(i)

X_text_clean = np.delete(X_text, del_index, axis=1)

X_text_clean[X_text_clean == ''] = '0.0'

X = X_text_clean.astype(float)

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

y = np.where(y_text == 'BENIGN', 0, 1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Rozkład klas przed balansem: {np.bincount(y_train)}")

rus = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

print(f"Rozkład klas po balansie: {np.bincount(y_train_resampled)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

models = {
    "Gaussian Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42)
}

print("\n--- Rozpoczynam trening modeli ---")

for name, model in models.items():
    print(f"\nModel: {name}")
    
    start_train = time.time()
    model.fit(X_train_scaled, y_train_resampled)
    train_time = time.time() - start_train
    
    start_predict = time.time()
    y_pred = model.predict(X_test_scaled)
    predict_time = time.time() - start_predict
    
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"Czas treningu:   {train_time:.4f} s")
    print(f"Czas predykcji:  {predict_time:.4f} s")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"F1-Score:        {f1:.4f}")

print("\nGenerowanie wykresu Ważności Cech (Feature Importance)...")

with open('small_DrDoS_DNS.csv', 'r') as f:
    header = f.readline().strip().split(',')

labels = header[:-1]

clean_labels = [name for i, name in enumerate(labels) if i not in del_index]

tree = DecisionTreeClassifier(max_depth=10, random_state=42)
tree.fit(X_train_scaled, y_train_resampled)

feature_importances = tree.feature_importances_

sorted_indecies = np.argsort(feature_importances)[::-1]
top_10_fi = feature_importances[sorted_indecies][:10]
top_10_names = [clean_labels[i] for i in sorted_indecies][:10]

plt.figure(figsize=(10, 6))
plt.barh(top_10_names[::-1], top_10_fi[::-1], color='darkred')
plt.xlabel('Ważność Cechy (Wpływ na decyzję modelu)')
plt.title('Top 10 cech decydujących o wykryciu ataku DDoS (Decision Tree)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig('feature_importance.png', dpi=300)
print("Wykres zapisano jako 'feature_importance.png'!")