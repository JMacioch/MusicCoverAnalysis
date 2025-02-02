import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

file_path_hist_rgb = r'...'
file_path_avg = r'...'
file_path_hist_hsv = r'...'


mode = 'combined_rgb_hist_avg'

if mode == 'rgb_hist' or mode == 'dominant_rgb':
    data_hist = np.load(file_path_hist_rgb, allow_pickle=True)
    X = data_hist['data']
    y = data_hist['labels']
    image_names = data_hist['image_names']

elif mode == 'avg_rgb':
    data_avg = np.load(file_path_avg, allow_pickle=True)
    X = data_avg['data']
    y = data_avg['labels']
    image_names = data_avg['image_names']

elif mode == 'hsv_hist' or mode == 'dominant_hsv':
    data = np.load(file_path_hist_hsv, allow_pickle=True)
    X = data['data']
    y = data['labels']
    image_names = data['image_names']

elif mode == 'combined_rgb_hist_avg':
    data_hist = np.load(file_path_hist_rgb, allow_pickle=True)
    data_avg = np.load(file_path_avg, allow_pickle=True)
    X_hist = data_hist['data']
    X_avg = data_avg['data']
    y = data_hist['labels']
    image_names = data_hist['image_names']
    X = np.concatenate((X_hist, X_avg), axis=1)


def sort_by_percentage(X, num_values=3):
    sorted_features = []
    for features in X:
        values = []
        percentages = []
        for i in range(0, len(features), num_values + 1):
            values.append(features[i:i + num_values])
            percentages.append(features[i + num_values])
        sorted_indices = np.argsort(percentages)[::-1]
        sorted_values = []
        for idx in sorted_indices:
            sorted_values.extend(values[idx])
            sorted_values.append(percentages[idx])

        sorted_features.append(sorted_values)
    return np.array(sorted_features)


def normalize_rgb_percentage(X):
    X_normalized = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(0, X.shape[1], 4):
            X_normalized[i, j] = X[i, j] / 255.0
            X_normalized[i, j + 1] = X[i, j + 1] / 255.0
            X_normalized[i, j + 2] = X[i, j + 2] / 255.0
            X_normalized[i, j + 3] = X[i, j + 3] / 100.0
    return X_normalized


def normalize_hsv_percentage(X):
    X_normalized = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(0, X.shape[1], 4):
            X_normalized[i, j] = X[i, j] / 360.0
            X_normalized[i, j + 1] = X[i, j + 1] / 100.0
            X_normalized[i, j + 2] = X[i, j + 2] / 100.0
            X_normalized[i, j + 3] = X[i, j + 3] / 100.0
    return X_normalized


def normalize_avg_rgb(X):
    return X / 255.0


def extract_dominant_color_rgb(X):
    dominant_colors = []
    for features in X:
        percentages = features[3::4]
        dominant_index = np.argmax(percentages)
        dominant_color = features[dominant_index * 4:dominant_index * 4 + 3]
        dominant_colors.append(dominant_color)
    return np.array(dominant_colors)


def extract_dominant_color_hsv(X):
    dominant_colors = []
    for features in X:
        percentages = features[3::4]
        dominant_index = np.argmax(percentages)
        dominant_color = features[dominant_index * 4:dominant_index * 4 + 3]
        dominant_colors.append(dominant_color)
    return np.array(dominant_colors)


def normalize_dominant_hsv(X):
    X_normalized = np.zeros_like(X)
    for i in range(X.shape[0]):
        X_normalized[i, 0] = X[i, 0] / 360.0
        X_normalized[i, 1] = X[i, 1] / 100.0
        X_normalized[i, 2] = X[i, 2] / 100.0
    return X_normalized


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


if mode == 'rgb_hist':
    X_sorted = sort_by_percentage(X, num_values=3)
    X_sorted_normalized = normalize_rgb_percentage(X_sorted)
elif mode == 'avg_rgb':
    X_sorted_normalized = normalize_avg_rgb(X)
elif mode == 'hsv_hist':
    X_sorted = sort_by_percentage(X, num_values=3)
    X_sorted_normalized = normalize_hsv_percentage(X_sorted)
elif mode == 'dominant_rgb':
    X_dominant = extract_dominant_color_rgb(X)
    X_sorted_normalized = normalize_avg_rgb(X_dominant)
elif mode == 'combined_rgb_hist_avg':
    X_hist_sorted = sort_by_percentage(X_hist, num_values=3)
    X_hist_normalized = normalize_rgb_percentage(X_hist_sorted)
    X_avg_normalized = normalize_avg_rgb(X_avg)
    X_sorted_normalized = np.concatenate((X_hist_normalized, X_avg_normalized), axis=1)
elif mode == 'dominant_hsv':
    X_dominant = extract_dominant_color_hsv(X)
    X_sorted_normalized = normalize_dominant_hsv(X_dominant)
le = LabelEncoder()
y = le.fit_transform(y)


def optuna_param(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'random_state': 40,
        'objective': 'multi:softmax',
        'num_class': len(np.unique(y)),
        'n_jobs': -1
    }

    gbm = xgb.XGBClassifier(**param)
    X_train, X_test, y_train, y_test = train_test_split(X_sorted_normalized, y, test_size=0.2, random_state=42, stratify=y)
    gbm.fit(X_train, y_train)

    y_pred = gbm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


study = optuna.create_study(direction='maximize')
study.optimize(optuna_param, n_trials=50)
print("Najlepsze hiperparametry: ", study.best_params)

best_params = study.best_params
best_model = xgb.XGBClassifier(**best_params, objective='multi:softmax', num_class=len(np.unique(y)), random_state=40, n_jobs=-1)
X_train, X_test, y_train, y_test = train_test_split(X_sorted_normalized, y, test_size=0.2, random_state=42, stratify=y)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
plot_confusion_matrix(y_test, y_pred, le.classes_)
