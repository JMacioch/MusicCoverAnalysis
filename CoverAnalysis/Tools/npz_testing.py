import numpy as np


def find_max_value_in_features(features):
    max_value = np.max(features)
    max_position = np.unravel_index(np.argmax(features), features.shape)
    return max_value, max_position


def load_npz_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    features = data['data']
    labels = data['labels']
    image_names = data['image_names']

    return features, labels, image_names


def display_max_value(features, labels, image_names):
    max_value, max_position = find_max_value_in_features(features)

    image_index = max_position[0]
    #print(f"Biggest feature value: {max_value}")
    #print(f"Index: {image_index}")
    #print(f"Label: {labels[image_index]}")
    #print(f"Image Name: {image_names[image_index]}")



if __name__ == '__main__':
    file_path = r'...'
    features, labels, image_names = load_npz_data(file_path)
    display_max_value(features, labels, image_names)