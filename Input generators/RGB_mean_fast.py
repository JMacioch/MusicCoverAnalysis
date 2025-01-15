import cv2
import numpy as np
import os
import time
from matplotlib import pyplot as plt

np.set_printoptions(suppress=True)


def create_weight_map(shape):
    h, w = shape[:2]
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2
    dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_dist = dist_from_center.max()
    normalized_dist = dist_from_center / max_dist
    weight_map = 1 + (1 - normalized_dist)
    return weight_map


def get_weighted_average_color(image, weight_map):
    image = image.astype('float32')
    weighted_image = np.zeros(image.shape)
    for i in range(3):
        weighted_image[:, :, i] = image[:, :, i] * weight_map

    total_weight = np.sum(weight_map)
    avg_color = np.sum(weighted_image, axis=(0, 1)) / total_weight

    return avg_color.astype("uint8").tolist()


def process_images_in_folder(data_folder):
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
                weight_map = create_weight_map(image_rgb.shape[:2])
                avg_color = get_weighted_average_color(image_rgb, weight_map)

                data.append(avg_color)
                image_names.append(image_file)
                labels.append(genre)

                image_count += 1
                if image_count % 100 == 0:
                    print(f"Processed: {image_count}")

    return np.array(data, dtype=object), np.array(labels), np.array(image_names)


def save_to_npz(filename, data, labels, image_names):
    np.savez_compressed(filename, data=data, labels=labels, image_names=image_names)
    print(f"Data saved {filename}")


if __name__ == '__main__':
    data_folder = '...'
    X, y, image_names = process_images_in_folder(data_folder)

    print("Feature matrix shape:", X.shape)
    #print("X (średnie kolory):", X)
    #print("Image names:", image_names)
    # print("Labels:", y)

    save_to_npz('average_color_data.npz', X, y, image_names)
