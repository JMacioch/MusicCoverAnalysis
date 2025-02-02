import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
import time
from sklearn.exceptions import ConvergenceWarning
import warnings

np.set_printoptions(suppress=True)


def centroidHistogram(clt, weight_map, image_shape):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    hist = np.zeros(len(numLabels) - 1)

    for idx, label in enumerate(clt.labels_):
        y = idx // image_shape[1]
        x = idx % image_shape[1]

        hist[label] += weight_map[y, x]

    hist = hist.astype("float")
    hist /= hist.sum()
    return hist


def create_weight_map(shape):
    h, w = shape[:2]
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2
    dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    max_dist = dist_from_center.max()
    normalized_dist = dist_from_center / max_dist

    weight_map = 1 + (1 - normalized_dist)
    return weight_map


def process_images_in_folder(data_folder, num_clusters=5):
    data = []
    labels = []
    image_names = []
    genres = ["blues", "classical", "country", "electronic", "hip-hop",
              "jazz", "metal", "pop", "reggae", "rock"]

    image_count = 0

    for genre in genres:
        genre_folder = os.path.join(data_folder, genre)
        for image_file in os.listdir(genre_folder):
            image_path = os.path.join(genre_folder, image_file)
            if image_path.endswith(('.png', '.jpg', '.jpeg')):
                start_time = time.time()

                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_flatten = image_rgb.reshape((image_rgb.shape[0] * image_rgb.shape[1], 3))

                weight_map = create_weight_map(image_rgb.shape[:2])
                clt = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    clt.fit(image_flatten)

                    if len(np.unique(clt.labels_)) < num_clusters:
                        print(f"Found one cluster: {image_file}")
                        first_pixel_rgb = image_rgb[0, 0].astype("uint8")
                        features = list(first_pixel_rgb)
                        features.append(100.0)

                        for _ in range(num_clusters - 1):
                            features.extend([0, 0, 0])
                            features.append(0.0)

                        data.append(features)
                        labels.append(genre)
                        image_names.append(image_file)

                        continue

                hist = centroidHistogram(clt, weight_map, image_rgb.shape)
                features = []
                for i in range(num_clusters):
                    rgb = np.round(clt.cluster_centers_[i]).astype("int")
                    percentage = round(hist[i] * 100, 2)
                    features.extend(rgb.tolist())
                    features.append(percentage)

                data.append(features)
                labels.append(genre)
                image_names.append(image_file)

                image_count += 1
                if image_count % 100 == 0:
                    print(f"Processed {image_count} images")

    return np.array(data, dtype=object), np.array(labels), np.array(image_names)


def save_to_npz(filename, data, labels, image_names):
    np.savez_compressed(filename, data=data, labels=labels, image_names=image_names)
    print(f"Data saved {filename}")


if __name__ == '__main__':
    data_folder = '...'
    X, y, image_names = process_images_in_folder(data_folder, num_clusters=5)

    print("Feature matrix shape:", X.shape)
    # print("X:", X)
    # print("Labels:", y)
    # print("Image names:", image_names)

    save_to_npz('../image_data.npz', X, y, image_names)