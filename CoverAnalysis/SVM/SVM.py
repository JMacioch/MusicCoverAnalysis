import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

file_path_hist_rgb = r'...'
file_path_avg = r'...'
file_path_hist_hsv = r'...'


mode = 'rgb_hist'
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

class_weights = {
    'blues': 1,
    'classical': 1,
    'country': 1,
    'electronic': 1,
    'hip-hop': 1,
    'jazz': 1,
    'metal': 1,
    'pop': 1,
    'reggae': 1,
    'rock': 1
}


def normalize_rgb_percentage(X):
    X_normalized = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(0, X.shape[1], 4):
            X_normalized[i, j] = X[i, j] / 255.0
            X_normalized[i, j+1] = X[i, j+1] / 255.0
            X_normalized[i, j+2] = X[i, j+2] / 255.0
            X_normalized[i, j+3] = X[i, j+3] / 100.0
    return X_normalized


def normalize_hsv_percentage(X):
    X_normalized = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(0, X.shape[1], 4):
            X_normalized[i, j] = X[i, j] / 360.0
            X_normalized[i, j+1] = X[i, j+1] / 100.0
            X_normalized[i, j+2] = X[i, j+2] / 100.0
            X_normalized[i, j+3] = X[i, j+3] / 100.0
    return X_normalized


def normalize_avg_rgb(X):
    return X / 255.0


def extract_dominant_color_rgb(X):
    dominant_colors = []
    for features in X:
        percentages = features[3::4]
        dominant_index = np.argmax(percentages)
        dominant_color = features[dominant_index*4:dominant_index*4+3]
        dominant_colors.append(dominant_color)
    return np.array(dominant_colors)


def extract_dominant_color_hsv(X):
    dominant_colors = []
    for features in X:
        percentages = features[3::4]
        dominant_index = np.argmax(percentages)
        dominant_color = features[dominant_index*4:dominant_index*4+3]
        dominant_colors.append(dominant_color)
    return np.array(dominant_colors)


def normalize_dominant_hsv(X):
    X_normalized = np.zeros_like(X)
    for i in range(X.shape[0]):
        X_normalized[i, 0] = X[i, 0] / 360.0
        X_normalized[i, 1] = X[i, 1] / 100.0
        X_normalized[i, 2] = X[i, 2] / 100.0
    return X_normalized


def sort_by_percentage(X, num_values=3):
    sorted_features = []
    for features in X:
        values = []
        percentages = []
        for i in range(0, len(features), num_values + 1):
            values.append(features[i:i+num_values])
            percentages.append(features[i+num_values])

        sorted_indices = np.argsort(percentages)[::-1]
        sorted_values = []
        for idx in sorted_indices:
            sorted_values.extend(values[idx])
            sorted_values.append(percentages[idx])

        sorted_features.append(sorted_values)
    return np.array(sorted_features)


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

for i in range(2):
    print(f"Image Name: {image_names[i]}")
    print(f"Label: {y[i]}")
    print(f"Processed Values: {X_sorted_normalized[i]}")
    print('-' * 50)


X_train, X_test, y_train, y_test = train_test_split(X_sorted_normalized, y, test_size=0.2, random_state=42, stratify=y)

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1]
}

grid = GridSearchCV(SVC(class_weight=class_weights), param_grid, refit=True, verbose=2, return_train_score=True, n_jobs=5)

grid.fit(X_train, y_train)

means = grid.cv_results_['mean_test_score']
params = grid.cv_results_['params']
for mean, param in zip(means, params):
    print(f"{param}: accueacy = {mean:.4f}")

print("\nNajlepsze parametry:", grid.best_params_)

y_pred = grid.predict(X_test)

print("\nAccuracy na zbiorze testowym:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
