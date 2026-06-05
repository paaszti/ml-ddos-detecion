import numpy as np
import time
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline 
import matplotlib.pyplot as plt

FILE = "../multiclass_dataset_perfect.csv"

raw_data = np.genfromtxt(FILE, delimiter=',', dtype=str, skip_header=1)
y_text = raw_data[:, -1]
X_text = raw_data[:, :-1]

with open(FILE, 'r') as f:
    header = f.readline().strip().split(',')

banned_columns = [
    'Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port', 
    'Destination IP', 'Destination Port', 'Timestamp', 
    'SimillarHTTP', 'Inbound'           
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

y_text_clean_list = []

for label in y_text:
    clean_label = label.strip()
    y_text_clean_list.append(clean_label)

y_text_clean = np.array(y_text_clean_list)

le = LabelEncoder()
y = le.fit_transform(y_text_clean)

models = {
    "Gaussian Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42)
}
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=67)

for name, model in models.items():
    print(f"\nModel: {name}")
    
    precision_list = []
    recall_list = []
    f1_list = []
    bal_acc_list = []
    
    for train_index, test_index in rskf.split(X, y):        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        
        precision_list.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
        recall_list.append(recall_score(y_test, y_pred, average='macro'))
        f1_list.append(f1_score(y_test, y_pred, average='macro'))
        bal_acc_list.append(balanced_accuracy_score(y_test, y_pred))
        

    print(f"Precision: {np.mean(precision_list):.4f} (± {np.std(precision_list):.4f})")
    print(f"Recall: {np.mean(recall_list):.4f} (± {np.std(recall_list):.4f})")
    print(f"F1-Score: {np.mean(f1_list):.4f} (± {np.std(f1_list):.4f})")
    print(f"Balanced Accuracy Score:  {np.mean(bal_acc_list):.4f} (± {np.std(bal_acc_list):.4f})")