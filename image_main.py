import os
from PIL import Image
import numpy as np
from perceptron import MultiClassPerceptron
import matplotlib.pyplot as plt


WIDTH = 50
HEIGHT = 50
LABEL_NAMES = ["NORMAL", "PNEUMONIA"]

def resize_images_helper(orig_path, save_path, name):
    print("resizing images")

    orig_dir = os.listdir(orig_path)
    i = 0
    for file in orig_dir:
        if file == '.DS_Store':
            continue
        elif os.path.isfile(orig_path + file):
            img = Image.open(orig_path + file)
            img_resized = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
            img_resized.save(save_path + name + str(i) + ".jpg", "JPEG", quality=95)
            if i != 0 and i % 250 == 0:
                print("progress: " + str(i) + " images processed")
            i += 1


def resize_images():
    resize_images_helper("./chest_xray/train/NORMAL/", "./data/train/NORMAL/", "norm-train-")
    resize_images_helper("./chest_xray/train/PNEUMONIA/", "./data/train/PNEUMONIA/", "pneu-train-")
    resize_images_helper("./chest_xray/test/NORMAL/", "./data/test/NORMAL/", "norm-test-")
    resize_images_helper("./chest_xray/test/PNEUMONIA/", "./data/test/PNEUMONIA/", "pneu-test-")


def load_data_helper(dir):
    print("loading data")

    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

    dataset = np.ndarray(shape=(len(files), WIDTH * HEIGHT))
    i = 0
    for _file in files:
        img = Image.open(dir + "/" + _file)
        img = img.convert("L")  # convert image to greyscale
        # img.show()
        dataset[i] = list(img.getdata())
        i += 1

    return dataset


def load_data():
    norm_train = load_data_helper("./data/train/NORMAL/")
    pneu_train = load_data_helper("./data/train/PNEUMONIA/")
    norm_test = load_data_helper("./data/test/NORMAL/")
    pneu_test = load_data_helper("./data/test/PNEUMONIA/")

    return norm_train, pneu_train, norm_test, pneu_test


# def plot_visualization(images, classes, cmap):
#
#     fig, ax = plt.subplots(2, figsize=(5, 5))
#     for i in range(len(LABEL_NAMES)):
#         ax[i%2].imshow(images.reshape((50, 50)), cmap=cmap)
#         ax[i%2].set_xticks([])
#         ax[i%2].set_yticks([])
#         ax[i%2].set_title(classes[i])
#     plt.show()

def plot_visualization(images, cmap):
    """Plot the visualizations
    """
    fig, ax = plt.subplots(2, figsize=(12, 5))
    for i in range(len(LABEL_NAMES)):
        ax[i%2].imshow(images[:, i].reshape((50, 50)), cmap=cmap)
        ax[i%2].set_xticks([])
        ax[i%2].set_yticks([])
        ax[i%2].set_title(LABEL_NAMES[i])
    plt.show()


if __name__ == '__main__':

    # resize_images()

    norm_train, pneu_train, norm_test, pneu_test = load_data()

    feature_dim = WIDTH * HEIGHT

    perceptron = MultiClassPerceptron(feature_dim)
    perceptron.train(norm_train, pneu_train)

    print(perceptron.weights)

    # ??: not sure if this is plotting just the final weights or actually
    # is evaluating for each label..
    plot_visualization(perceptron.weights[1:, :], None)

