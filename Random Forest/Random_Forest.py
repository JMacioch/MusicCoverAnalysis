from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

file_path_hist_rgb = r'...'
file_path_avg = r'...'
file_path_hist_hsv = r'...'

mode = 'combined_rgb_hist_avg'

if mode == 'dominant_hsv' or mode == 'dominant_rgb':
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


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


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


def normalize_avg_rgb(X):
    return X / 255.0


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

X_train, X_test, y_train, y_test = train_test_split(X_sorted_normalized, y, test_size=0.2, random_state=40, stratify=y)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [3, 5, 7],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=40, class_weight='balanced')
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("Best param: ", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

unique_labels = np.unique(y)
plot_confusion_matrix(y_test, y_pred, unique_labels)
