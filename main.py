import numpy as np
import csv
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline 

def run_evaluation(banned_keywords, experiment_name, results_file, importances_file, cm_file):
    file_name = "./assets/multiclass_dataset_4k.csv"
    
    raw_data = np.genfromtxt(file_name, delimiter=',', dtype=str, skip_header=1)
    y_text = raw_data[:, -1]
    x_text = raw_data[:, :-1]
    
    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
    delete_indices = []
    feature_names = []
    current_index = 0
    
    for col_name in header[:-1]:
        col_name_lower = col_name.lower()
        match = False
        for banned in banned_keywords:
            if banned in col_name_lower:
                delete_indices.append(current_index)
                match = True
                break
        if not match:
            feature_names.append(col_name.strip())
        current_index += 1
        
    x_text_clean = np.delete(x_text, delete_indices, axis=1)
    x_text_clean[x_text_clean == ''] = '0.0'
    
    x_array = x_text_clean.astype(float)
    x_array = np.nan_to_num(x_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    y_text_clean_list = []
    for label in y_text:
        clean_label = label.strip()
        y_text_clean_list.append(clean_label)
        
    y_text_clean = np.array(y_text_clean_list)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_text_clean)
    class_names = label_encoder.classes_
    
    pipeline_dt = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier(max_depth=10, random_state=42))
    ])
    
    pipeline_dt.fit(x_array, y_encoded)
    model_dt = pipeline_dt.named_steps['classifier']
    importances = model_dt.feature_importances_
    
    with open(importances_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for index in range(len(feature_names)):
            row_to_write = []
            row_to_write.append(experiment_name)
            row_to_write.append(feature_names[index])
            row_to_write.append(str(importances[index]))
            writer.writerow(row_to_write)
    
    models = {}
    models["Gaussian Naive Bayes"] = GaussianNB()
    models["K-Nearest Neighbors"] = KNeighborsClassifier(n_neighbors=5)
    models["Decision Tree"] = DecisionTreeClassifier(max_depth=10, random_state=42)
    
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=67)
    
    print(f"\n--- STARTING EXPERIMENT: {experiment_name} ---")
    
    for model_name, model in models.items():
        print(f"\nEvaluating: {model_name}")
        
        precision_list = []
        recall_list = []
        f1_list = []
        bal_acc_list = []
        
        num_classes = len(class_names)
        total_cm = np.zeros((num_classes, num_classes), dtype=int)
        
        for train_index, test_index in rskf.split(x_array, y_encoded):
            x_train = x_array[train_index]
            x_test = x_array[test_index]
            y_train = y_encoded[train_index]
            y_test = y_encoded[test_index]
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
            
            pipeline.fit(x_train, y_train)
            y_pred = pipeline.predict(x_test)
            
            precision_val = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall_val = recall_score(y_test, y_pred, average='macro')
            f1_val = f1_score(y_test, y_pred, average='macro')
            bal_acc_val = balanced_accuracy_score(y_test, y_pred)
            
            precision_list.append(precision_val)
            recall_list.append(recall_val)
            f1_list.append(f1_val)
            bal_acc_list.append(bal_acc_val)
            
            fold_cm = confusion_matrix(y_test, y_pred)
            total_cm = total_cm + fold_cm
            
        mean_precision = np.mean(precision_list)
        std_precision = np.std(precision_list)
        mean_recall = np.mean(recall_list)
        std_recall = np.std(recall_list)
        mean_f1 = np.mean(f1_list)
        std_f1 = np.std(f1_list)
        mean_bal_acc = np.mean(bal_acc_list)
        std_bal_acc = np.std(bal_acc_list)
        
        with open(results_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            data_row = []
            data_row.append(experiment_name)
            data_row.append(model_name)
            data_row.append(str(mean_precision))
            data_row.append(str(std_precision))
            data_row.append(str(mean_recall))
            data_row.append(str(std_recall))
            data_row.append(str(mean_f1))
            data_row.append(str(std_f1))
            data_row.append(str(mean_bal_acc))
            data_row.append(str(std_bal_acc))
            writer.writerow(data_row)
            
        with open(cm_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for i in range(num_classes):
                for j in range(num_classes):
                    row_to_write = []
                    row_to_write.append(experiment_name)
                    row_to_write.append(model_name)
                    row_to_write.append(class_names[i])
                    row_to_write.append(class_names[j])
                    row_to_write.append(str(total_cm[i][j]))
                    writer.writerow(row_to_write)
            
        print(f"Precision: {mean_precision:.3f} (+/- {std_precision:.3f})")
        print(f"Recall: {mean_bal_acc:.3f} (+/- {std_bal_acc:.3f})")
        print(f"F1-Score: {mean_f1:.3f} (+/- {std_f1:.3f})")
        print(f"Recall: {mean_recall:.3f} (+/- {std_recall:.3f})")

def execute_all_experiments():
    results_csv_file = './assets/results/combined_metrics_results.csv'
    importances_csv_file = './assets/results/combined_feature_importances.csv'
    cm_csv_file = './assets/results/combined_confusion_matrices.csv'
    
    with open(results_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header_row = []
        header_row.append('experiment_type')
        header_row.append('classifier')
        header_row.append('precision_mean')
        header_row.append('precision_std')
        header_row.append('recall_mean')
        header_row.append('recall_std')
        header_row.append('f1_score_mean')
        header_row.append('f1_score_std')
        header_row.append('balanced_accuracy_mean')
        header_row.append('balanced_accuracy_std')
        writer.writerow(header_row)
        
    with open(importances_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header_row = []
        header_row.append('experiment_type')
        header_row.append('feature_name')
        header_row.append('importance_weight')
        writer.writerow(header_row)
        
    with open(cm_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header_row_cm = []
        header_row_cm.append('experiment_type')
        header_row_cm.append('classifier')
        header_row_cm.append('true_label')
        header_row_cm.append('predicted_label')
        header_row_cm.append('count')
        writer.writerow(header_row_cm)
        
    banned_keywords_clean = []
    banned_keywords_clean.append('unnamed')
    banned_keywords_clean.append('flow id')
    banned_keywords_clean.append('source ip')
    banned_keywords_clean.append('source port')
    banned_keywords_clean.append('destination ip')
    banned_keywords_clean.append('destination port')
    banned_keywords_clean.append('timestamp')
    banned_keywords_clean.append('simillarhttp')
    banned_keywords_clean.append('inbound')
    
    run_evaluation(banned_keywords_clean, "CLEAN_DATA", results_csv_file, importances_csv_file, cm_csv_file)
    
    banned_keywords_leak = []
    banned_keywords_leak.append('unnamed')
    banned_keywords_leak.append('flow id')
    banned_keywords_leak.append('source ip')
    banned_keywords_leak.append('destination ip')
    banned_keywords_leak.append('timestamp')
    banned_keywords_leak.append('simillarhttp')
    
    run_evaluation(banned_keywords_leak, "DATA_LEAK", results_csv_file, importances_csv_file, cm_csv_file)

if __name__ == "__main__":
    execute_all_experiments()