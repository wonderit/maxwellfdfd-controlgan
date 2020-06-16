import numpy as np
import pandas as pd
import pickle
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

DATASETS_TRAIN = [
    'binary_501',
    'binary_502',
    'binary_503',
    'binary_504',
    'binary_505',
    'binary_506',
    'binary_507',
    'binary_508',
    'binary_509',
    'binary_510',
    'binary_511',
    'binary_512',
    'binary_1001',
    'binary_1002',
    'binary_1003',
    'binary_rl_fix_501',
    'binary_rl_fix_502',
    'binary_rl_fix_503',
    'binary_rl_fix_504',
    'binary_rl_fix_505',
    'binary_rl_fix_506',
    'binary_rl_fix_507',
    'binary_rl_fix_508',
    'binary_rl_fix_509',
    'binary_rl_fix_510',
    'binary_rl_fix_511',
    'binary_rl_fix_512',
    'binary_rl_fix_513',
    'binary_rl_fix_514',
    'binary_rl_fix_515',
    'binary_rl_fix_516',
    'binary_rl_fix_517',
    'binary_rl_fix_518',
    'binary_rl_fix_519',
    'binary_rl_fix_520',
    'binary_rl_fix_1001',
    'binary_rl_fix_1002',
    'binary_rl_fix_1003',
    'binary_rl_fix_1004',
    'binary_rl_fix_1005',
    'binary_rl_fix_1006',
    'binary_rl_fix_1007',
    'binary_rl_fix_1008',
]

DATASETS_VALID = [
    'binary_1004',
    'binary_test_1001',
    'binary_test_1002',
    'binary_rl_fix_1009',
    'binary_rl_fix_1010',
    'binary_rl_fix_1011',
    'binary_rl_fix_1012',
    'binary_rl_fix_1013',
    'binary_rl_fix_test_1001',
]

DATASETS_TEST = [
    'binary_new_test_501',
    'binary_new_test_1501',
    'binary_rl_fix_1014',
    'binary_rl_fix_1015',
    'binary_rl_fix_test_1002',
    'binary_rl_fix_test_1003',
    'binary_rl_fix_test_1004',
    'binary_rl_fix_test_1005',
    'binary_test_1101',
]


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict['data'], dict['labels']


def compress_image(prev_image, n):
    height = prev_image.shape[0] // n
    width = prev_image.shape[1] // n
    new_image = np.zeros((height, width), dtype="uint8")
    for i in range(0, height):
        for j in range(0, width):
            new_image[i, j] = prev_image[n * i, n * j]
    return new_image


def cem_generator(data_type, batch_size, data_dir, is_regression=False):
    if data_type == 'train':
        DATAPATH = os.path.join(data_dir, 'train')
        DATASETS = DATASETS_TRAIN
    elif data_type == 'valid':
        DATAPATH = os.path.join(data_dir, 'valid')
        DATASETS = DATASETS_VALID
    else:
        DATAPATH = os.path.join(data_dir, 'test')
        DATASETS = DATASETS_TEST

    all_data = []
    all_labels = []
    print('data loading ... ')

    # load Train dataset
    for data in DATASETS:
        dataframe = pd.read_csv(os.path.join(DATAPATH, '{}.csv'.format(data)), delim_whitespace=False, header=None)
        dataset = dataframe.values

        # split into input (X) and output (Y) variables
        fileNames = dataset[:, 0]

        # 1. first try max
        all_labels.extend(dataset[:, 1:25])
        for idx, file in enumerate(fileNames):

            try:
                image = Image.open(os.path.join(DATAPATH, data, '{}.tiff'.format(int(file))))
                image = np.array(image, dtype=np.uint8)
            except (TypeError, FileNotFoundError) as te:
                image = Image.open(os.path.join(DATAPATH, data, '{}.tiff'.format(idx + 1)))
                try:
                    image = np.array(image, dtype=np.uint8)
                except:
                    continue
            image = compress_image(image, 5)

            all_data.append(np.array(image).flatten(order='C'))


    all_data = np.array(all_data)
    all_labels = np.array(all_labels)

    # Calculate transmittance
    all_labels /= 2767.

    # change 0 ~ 1 -> 0 ~ 255
    all_data *= 255

    # calculate argmax
    all_labels_df = pd.DataFrame(all_labels)
    if not is_regression:
        all_labels_df = all_labels_df.apply(lambda x: np.argmax(x), axis=1)

    # Countplot
    # y_train_df_for_countplot = pd.DataFrame(all_labels_df)
    # y_train_df_for_countplot.columns = ['wavelength (nm)']
    # sns.countplot(x="wavelength (nm)", data=y_train_df_for_countplot)
    # plt.title('Countplot for max transmittance wavelength')
    # plt.show()
    # exit()

    print('Data Loading... Train dataset Finished.')
    images = all_data
    labels = all_labels_df.values

    # Just use 12 classes
    filtered_index = labels > 11
    images = images[filtered_index]
    labels = labels[filtered_index]
    labels = labels - 12

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in range(int(len(images) / batch_size)):
            yield (images[i * batch_size:(i + 1) * batch_size], labels[i * batch_size:(i + 1) * batch_size])

    return get_epoch



def cifar_generator(filenames, batch_size, data_dir):
    all_data = []
    all_labels = []
    for filename in filenames:
        data, labels = unpickle(data_dir + '/' + filename)
        all_data.append(data)
        all_labels.append(labels)

    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in range(int(len(images) / batch_size)):
            yield (images[i * batch_size:(i + 1) * batch_size], labels[i * batch_size:(i + 1) * batch_size])

    return get_epoch


def load(batch_size, data_dir, is_regression=False):
    return (
        cem_generator('train', batch_size, data_dir, is_regression),
        cem_generator('test', batch_size, data_dir, is_regression)
    )


def main():
    batch_size = 10
    data_dir = '../cifar10'

    train_gen = cifar_generator(['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'], batch_size, data_dir)
    dev_gen = cifar_generator(['test_batch'], batch_size, data_dir)

    def inf_train_gen():
        while True:
            for images, _labels in train_gen():
                yield images, _labels

    gen = inf_train_gen()
    _data, _labels = next(gen)
    print(_data)
    print(_labels)

def cem():
    batch_size = 10
    data_dir = '../data'
    train_gen = cem_generator('train',  batch_size, data_dir)
    dev_gen = cem_generator('test', batch_size, data_dir)

    def inf_train_gen():
        while True:
            for images, _labels in train_gen():
                yield images, _labels

    gen = inf_train_gen()
    _data, _labels = next(gen)
    print(_data, _data.shape)
    print(_labels, _labels.shape)


if __name__ == "__main__":
    cem()
