import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import keras_tuner as kt
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


def load_input():
    file_path_hist = '../data_rgb_hist.npz'
    file_path_mean = '../data_rgb_mean.npz'
    data_hist = np.load(file_path_hist, allow_pickle=True)
    data_mean = np.load(file_path_mean, allow_pickle=True)
    image_names = data_hist['image_names']
    labels_identical = np.array_equal(data_hist['labels'], data_mean['labels'])
    image_names_identical = np.array_equal(data_hist['image_names'], data_mean['image_names'])
    sorted_hist_features = sort_histogram_by_percentage(data_hist['data'])

    if labels_identical and image_names_identical:
        combined_features = np.hstack((sorted_hist_features, data_mean['data']))
        combined_labels = data_mean['labels']
    else:
        combined_features = None

    return combined_features, combined_labels, image_names


def sort_histogram_by_percentage(features):
    sorted_features = features.copy().astype(np.float32)

    for i in range(features.shape[0]):
        rgb_percentage_pairs = []
        for j in range(0, len(features[i]), 4):
            rgb = features[i][j:j + 3]
            percentage = features[i][j + 3]
            rgb_percentage_pairs.append((rgb, percentage))

        rgb_percentage_pairs = sorted(rgb_percentage_pairs, key=lambda x: x[1], reverse=True)

        sorted_list = []
        for rgb, percentage in rgb_percentage_pairs:
            sorted_list.extend(rgb)
            sorted_list.append(percentage)

        sorted_features[i][:len(sorted_list)] = sorted_list

    return sorted_features


def normalize_features(features):
    normalized_features = features.copy().astype(np.float32)
    for i in range(features.shape[0]):
        for j in range(0, len(features[i]), 4):
            if j >= 20:
                normalized_features[i][j] = features[i][j] / 255.0
                normalized_features[i][j + 1] = features[i][j + 1] / 255.0
                normalized_features[i][j + 2] = features[i][j + 2] / 255.0
            else:
                normalized_features[i][j] = features[i][j] / 360.0
                normalized_features[i][j + 1] = features[i][j + 1] / 100.0
                normalized_features[i][j + 2] = features[i][j + 2] / 100.0

                normalized_features[i][j + 3] = features[i][j + 3] / 100.0

    return normalized_features


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def build_model(hp, num_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(23,)))
    model.add(tf.keras.layers.Dropout(0.1))

    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(tf.keras.layers.Dense(
            units=hp.Int(f'units_{i}', min_value=64, max_value=256, step=64),
            activation='relu',
            kernel_regularizer=l2(0.01)
        ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(hp.Float(f'dropout_{i}', min_value=0.2, max_value=0.5, step=0.1)))

    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd'])
    if optimizer_choice == 'adam':
        optimizer = Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]))
    else:
        optimizer = SGD(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]), momentum=0.9)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train_dnn_with_keras_tuner():
    features, labels, image_names = load_input()
    features_normalized = normalize_features(features)

    # print("Example image name:", image_names[0])
    # print("Normalized features for the example image:")
    # print(features_normalized[0])

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    num_classes = len(np.unique(labels_encoded))

    X_train, X_test, y_train, y_test = train_test_split(features_normalized, labels_encoded, test_size=0.2,
                                                        random_state=40,
                                                        stratify=labels_encoded)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    tuner = kt.Hyperband(
        lambda hp: build_model(hp, num_classes),
        objective='val_accuracy',
        max_epochs=100,
        factor=2,
        directory='my_dir',
        project_name='DNN_tuning'
    )

    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    tuner.search(X_train, y_train, epochs=50, validation_split=0.2, verbose=2,
                 callbacks=[lr_scheduler, early_stopping])

    best_model = tuner.get_best_models(num_models=1)[0]

    best_model.save('best_model_hist_rgb_mean_rgb.h5')

    best_model.summary()

    y_pred = best_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    print("Accuracy:", accuracy_score(y_test_classes, y_pred_classes))
    print("\nClassification Report:\n", classification_report(y_test_classes, y_pred_classes, target_names=le.classes_))

    plot_confusion_matrix(y_test_classes, y_pred_classes, le.classes_)


train_dnn_with_keras_tuner()
