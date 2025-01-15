import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import os
import time

np.set_printoptions(suppress=True)

def imshow(title="Image", image=None, ax=None):

    if ax is None:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB))
        plt.title(title)
        plt.axis('off')
    else:
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB))
        ax.set_title(title)
        ax.axis('off')


def create_weight_map(shape):
    h, w = shape[:2]
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2
    dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_dist = dist_from_center.max()
    normalized_dist = dist_from_center / max_dist
    weight_map = 1 + (1 - normalized_dist)
    return weight_map


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


def convert_hsv_to_standard(hsv_color):
    h, s, v = hsv_color
    h_standard = h * 2
    s_standard = (s / 255) * 100
    v_standard = (v / 255) * 100
    return [h_standard, s_standard, v_standard]


def plotColors(hist, centroids, ax=None):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    x_start = 0
    dominant_colors = []
    centroids_rgb = cv2.cvtColor(np.uint8([centroids]), cv2.COLOR_HSV2RGB)[0]

    for (percent, color_rgb, color_hsv) in zip(hist, centroids_rgb, centroids):
        end = x_start + (percent * 300)
        standard_hsv_color = convert_hsv_to_standard(color_hsv)
        h, s, v = standard_hsv_color

        cv2.rectangle(bar, (int(x_start), 0), (int(end), 50), color_rgb.tolist(), -1)
        x_start = end

        dominant_colors.append(([round(h, 2), round(s, 2), round(v, 2)], round(percent * 100, 2)))

    if ax is None:
        plt.imshow(bar)
        plt.title("Dominant colors (RGB)")
        plt.axis('off')

        text_position = -0.1
        for color, percent in dominant_colors:
            color_text = f"Color (HSV): H={color[0]:.2f}, S={color[1]:.2f}, V={color[2]:.2f}, {percent:.2f}%"
            plt.text(1.0, text_position, color_text, fontsize=10, color='black',
                     transform=plt.gca().transAxes, horizontalalignment='right', verticalalignment='top')
            text_position -= 0.2
    else:
        ax.imshow(bar)
        ax.set_title("Dominant colors (RGB)")
        ax.axis('off')

        text_position = -0.1
        for color, percent in dominant_colors:
            color_text = f"Color (HSV): H={color[0]:.2f}, S={color[1]:.2f}, V={color[2]:.2f}, {percent:.2f}%"
            ax.text(1.0, text_position, color_text, fontsize=10, color='black',
                    transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')
            text_position -= 0.2

    return dominant_colors

def process_images_in_folder(data_folder, num_clusters=5):
    data = []
    labels = []
    genres = ["blues", "classical", "country", "electronic", "hip-hop",
              "jazz", "metal", "pop", "reggae", "rock"]

    image_count = 0
    first_image = True

    for genre in genres:
        genre_folder = os.path.join(data_folder, genre)
        for image_file in os.listdir(genre_folder):
            image_path = os.path.join(genre_folder, image_file)
            if image_path.endswith(('.png', '.jpg', '.jpeg')):
                start_time = time.time()

                image = cv2.imread(image_path)
                image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                image_flatten = image_hsv.reshape((image_hsv.shape[0] * image_hsv.shape[1], 3))
                weight_map = create_weight_map(image_hsv.shape[:2])
                clt = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
                clt.fit(image_flatten)

                hist = centroidHistogram(clt, weight_map, image_hsv.shape)
                features = []
                for i in range(num_clusters):
                    hsv = clt.cluster_centers_[i].astype("uint8")
                    percentage = round(hist[i] * 100, 2)
                    features.extend(hsv.tolist())
                    features.append(percentage)

                data.append(features)
                labels.append(genre)

                image_count += 1
                if image_count % 100 == 0:
                    print(f"Processed: {image_count}")

                if first_image:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                    imshow("Input (RGB)", image_hsv, ax=ax1)
                    plotColors(hist, clt.cluster_centers_, ax=ax2)
                    plt.show(block=True)
                    first_image = False

    return np.array(data, dtype=object), np.array(labels)


def save_to_npz(filename, data, labels):
    np.savez_compressed(filename, data=data, labels=labels)
    print(f"Data saved {filename}")

if __name__ == '__main__':
    data_folder = '...'
    X, y = process_images_in_folder(data_folder, num_clusters=5)

    print("Feature matrix shape:", X.shape)
    save_to_npz('../image_data_hsv.npz', X, y)
