from subprocess import check_output
import os, sys
from PIL import Image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

WIDTH = 50
HEIGHT = 50

def resize_images_helper(orig_path, save_path, name):
    print("resizing images")

    orig_dir = os.listdir(orig_path)
    i = 0
    for file in orig_dir:
        if file == '.DS_Store':
            continue
        elif os.path.isfile(orig_path + file):
            img = Image.open(orig_path + file)
            img_resized = img.resize((WIDTH,HEIGHT), Image.ANTIALIAS)
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
        dataset[i] = list(img.getdata())
        i += 1

    return dataset

def load_data():
    norm_train = load_data_helper("./data/train/NORMAL/")
    pneu_train = load_data_helper("./data/train/PNEUMONIA/")
    norm_test = load_data_helper("./data/test/NORMAL/")
    pneu_test = load_data_helper("./data/test/PNEUMONIA/")

    return norm_train, pneu_train, norm_test, pneu_test


if __name__ == '__main__':

    # resize_images()

    norm_train, pneu_train, norm_test, pneu_test = load_data()






