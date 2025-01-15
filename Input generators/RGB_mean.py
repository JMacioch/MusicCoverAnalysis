import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_average_color(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    average_color_per_row = np.average(image, axis=0)
    average_color = np.average(average_color_per_row, axis=0)

    return average_color.astype("uint8").tolist()


def imshow(title="Image", image=None, ax=None):
    if ax is None:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    else:
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')

if __name__ == '__main__':
    image = cv2.imread('a.jpg')
    average_color = get_average_color(image)

    print("Average color (RGB):", average_color)

    color_image = np.zeros((100, 100, 3), dtype="uint8")
    color_image[:] = average_color

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    imshow("Input Image", image, ax=ax1)
    ax2.imshow(color_image)
    ax2.set_title(f"Average color (RGB): {average_color}")
    ax2.axis('off')
    plt.show()

    data_folder = '...'