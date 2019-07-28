from subprocess import check_output
import os, sys
from PIL import Image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



# convert jpegs into numpy arrays
def load_data():
    print(check_output(["ls", "./chest_xray/train"]).decode("utf8"))

    # norm_folder = "./chest_xray/train/NORMAL"

    folder = "./chest_xray/train/NORMAL"

    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    print("Working with {0} normal images".format(len(onlyfiles)))

    norm_train = []
    norm_files = []
    # y_train = []
    # i = 0
    for _file in onlyfiles:
        norm_files.append(_file)
        label_in_file = _file.find("_")
        # y_train.append(int(_file[0:label_in_file]))

    print("Files in train_files: %d" % len(norm_files))

    i = 0
    for _file in norm_files:
        img = load_img(folder + "/" + _file)  # this is a PIL image
        img.thumbnail((28,26))

        # # Convert to Numpy Array
        # x = img_to_array(img)

        x = np.array(img)

        # # x = x.reshape((3, 120, 160))
        #
        # # Normalize
        # x = (x - 128.0) / 128.0
        x.reshape((28*26,))
        norm_train.append(x)
        i += 1
        if i > 10:
            break
        # if i % 250 == 0:
        #     print("%d images to array" % i)

    norm_test = []
    pneu_train = []
    pneu_test = []

    # for image in norm_folder:
    #     img = Image.open(image)
    #     np.append(norm_train, img)



    return np.array(norm_train), np.array(norm_test), np.array(pneu_train), np.array(pneu_test);


def resize_images(orig_path, save_path, name):
    print("resizing images")

    orig_dir = os.listdir(orig_path)
    i = 0
    for file in orig_dir:
        if file == '.DS_Store':
            continue
        elif os.path.isfile(orig_path + file):
            img = Image.open(orig_path + file)
            img_resized = img.resize((50,50), Image.ANTIALIAS)
            img_resized.save(save_path + name + str(i) + ".jpg", "JPEG", quality=95)
            if i != 0 and i % 250 == 0:
                print("progress: " + str(i) + " images processed")
            i += 1



if __name__ == '__main__':
    print("loading x-rays")

    resize_images("./chest_xray/train/NORMAL/", "./data/train/NORMAL/", "norm-train-")
    resize_images("./chest_xray/train/PNEUMONIA/", "./data/train/PNEUMONIA/", "pneu-train-")
    resize_images("./chest_xray/test/NORMAL/", "./data/test/NORMAL/", "norm-test-")
    resize_images("./chest_xray/test/PNEUMONIA/", "./data/test/PNEUMONIA/", "pneu-test-")


    # norm_train, norm_test, pneu_train, pneu_test = load_data()


