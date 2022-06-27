import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
from keras import losses
from PIL import Image
import numpy as np
import argparse
import os
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import functools
import tflib as lib


class CustomLoss:
    def __init__(self, _loss_function):
        super(CustomLoss, self).__init__()
        self.loss_function_array = _loss_function.split(',')

    def tf_diff_axis_1(self, a):
        return a[:, 1:] - a[:, :-1]

    def tf_minmax_axis_1(self, a):
        b = self.tf_diff_axis_1(a)
        sign = K.sign(b)
        abs_sign = tf.abs(self.tf_diff_axis_1(sign))
        mask_array = K.greater(abs_sign, 0)

        result = tf.where(mask_array, a[:, 1:-1], tf.zeros_like(a[:, 1:-1]))

        return result

    def custom_loss(self, y_true, y_pred):
        loss = 0
        y_true_diff = self.tf_diff_axis_1(y_true)
        y_pred_diff = self.tf_diff_axis_1(y_pred)
        threshold_value = 0
        y_true_diff_binary = K.cast(K.greater(y_true_diff, threshold_value), K.floatx())
        y_pred_diff_binary = K.cast(K.greater(y_pred_diff, threshold_value), K.floatx())
        y_true_minmax = self.tf_minmax_axis_1(y_true)
        y_pred_minmax = self.tf_minmax_axis_1(y_pred)

        if 'mse' in self.loss_function_array:
            loss = loss + K.mean(K.square(y_pred - y_true))

        if 'diff_mse' in self.loss_function_array:
            loss = loss + K.mean(K.square(y_pred_diff - y_true_diff))

        if 'rmse' in self.loss_function_array:
            loss = loss + K.sqrt(K.mean(K.square(y_pred - y_true)))

        if 'diff_rmse' in self.loss_function_array:
            loss = loss + K.sqrt(K.mean(K.square(y_pred_diff - y_true_diff)))

        if 'diff_ce' in self.loss_function_array:
            loss = loss + losses.binary_crossentropy(y_true_diff, y_pred_diff)

        if 'diff_bce' in self.loss_function_array:
            loss = loss + losses.binary_crossentropy(y_true_diff_binary, y_pred_diff_binary)

        if 'diff_rmse_minmax' in self.loss_function_array:
            loss = loss + K.sqrt(K.mean(K.square(y_pred_minmax - y_true_minmax)))

        if 'diff_poly' in self.loss_function_array:
            x = np.arange(24)
            loss = loss + np.sum(
                (np.polyval(np.polyfit(x, y_pred, 3)) - np.polyval(np.polyfit(x, y_true, 3))) ** 2
            )

        return loss

def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        runningTime = time.time() - startTime_for_tictoc
        toc_ = "Elapsed time is " + str(runningTime) + " seconds."
        print(toc_)
        return runningTime
    else:
        toc_ = "Toc: start time not set"
        print(toc_)
        return toc_

def scale(arr, std, mean):
    arr -= mean
    arr /= (std + 1e-7)
    return arr


def rescale(arr, std, mean):
    arr = arr * std
    arr = arr + mean

    return arr


def compress_image(prev_image, n):
    height = prev_image.shape[0] // n
    width = prev_image.shape[1] // n
    new_image = np.zeros((height, width), dtype="uint8")
    for i in range(0, height):
        for j in range(0, width):
            new_image[i, j] = prev_image[n * i, n * j]
    return new_image

NUM_LABELS = 12
CONDITIONAL = True
ACGAN = True
NORMALIZATION_G = True  # Use batchnorm in generator? only t
NORMALIZATION_D = False  # Use batchnorm (or layernorm) in critic? only f
NORMALIZATION_C = True  # Use batchnorm (or layernorm) in classifier?t or f

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True, s_norm=False):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases,
                                   s_norm=s_norm)
    output = tf.add_n(
        [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True, s_norm=False):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.nn.depth_to_space(output, 2)
    output = tf.transpose(output, [0, 3, 1, 2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases,
                                   s_norm=s_norm)
    return output
def nonlinearity(x):
    return tf.nn.relu(x)
def Normalize(name, inputs, labels=None):
    """This is messy, but basically it chooses between batchnorm, layernorm,
    their conditional variants, or nothing, depending on the value of `name` and
    the global hyperparam flags."""
    if not CONDITIONAL:
        labels = None
    if CONDITIONAL and ACGAN and ('Discriminator' in name):
        labels = None
    if CONDITIONAL and ACGAN and ('Classifier' in name):
        labels = None

    if ('Discriminator' in name) and NORMALIZATION_D:
        return lib.ops.layernorm.Layernorm(name, [1, 2, 3], inputs, labels=labels, n_labels=NUM_LABELS)
    elif ('Classifier' in name) and NORMALIZATION_C:
        return lib.ops.layernorm.Layernorm(name, [1, 2, 3], inputs)
    elif ('Generator' in name) and NORMALIZATION_G:
        if labels is not None:
            # print('labels:', labels.dtype, labels)
            # labels = tf.cast(labels, tf.int32)
            return lib.ops.cond_batchnorm.Batchnorm(name, [0, 2, 3], inputs, labels=labels, n_labels=NUM_LABELS)
        else:
            return lib.ops.batchnorm.Batchnorm(name, [0, 2, 3], inputs, fused=True)
    else:
        return inputs
def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, no_dropout=False, labels=None,
                  s_norm=False):
    """
    resample: None, 'down', or 'up'
    """
    if resample == 'down':
        conv_1 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim, s_norm=s_norm)
        conv_2 = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim, s_norm=s_norm)
        conv_shortcut = functools.partial(ConvMeanPool, s_norm=s_norm)
    elif resample == 'up':
        conv_1 = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim, s_norm=s_norm)
        conv_shortcut = functools.partial(UpsampleConv, s_norm=s_norm)
        conv_2 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim, s_norm=s_norm)
    elif resample == None:
        conv_shortcut = functools.partial(lib.ops.conv2d.Conv2D, s_norm=s_norm)
        conv_1 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim, s_norm=s_norm)
        conv_2 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim, s_norm=s_norm)
    else:
        raise Exception('invalid resample value')

    if output_dim == input_dim and resample == None:
        shortcut = inputs  # Identity skip-connection
    else:
        shortcut = conv_shortcut(name + '.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(name + '.N1', output, labels=labels)
    output = nonlinearity(output)
    output = conv_1(name + '.Conv1', filter_size=filter_size, inputs=output)
    output = Normalize(name + '.N2', output, labels=labels)
    output = nonlinearity(output)
    output = conv_2(name + '.Conv2', filter_size=filter_size, inputs=output)

    return shortcut + output
def OptimizedResBlockClass1(inputs):
    conv_1 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=1, output_dim=32)
    conv_2 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=32, output_dim=32)
    conv_shortcut = functools.partial(lib.ops.conv2d.Conv2D)
    shortcut = conv_shortcut('Classifier.1.Shortcut', input_dim=1, output_dim=32, filter_size=1, he_init=False,
                             biases=True, inputs=inputs)

    output = inputs
    output = conv_1('Classifier.1.Conv1', filter_size=3, inputs=output)
    output = nonlinearity(output)
    output = conv_2('Classifier.1.Conv2', filter_size=3, inputs=output)
    return shortcut + output
def create_resnet_model(inputs, labels):
    output = tf.reshape(inputs, [-1, 1, 20, 40])
    output = OptimizedResBlockClass1(output)
    output = ResidualBlock('Classifier.2', 32, 32, 3, output, resample='down', labels=labels)
    # output = ResidualBlock('Classifier.3', 32, 32, 3, output, resample='down', labels=labels)
    # output = ResidualBlock('Classifier.4', 32, 64, 3, output, resample=None, labels=labels)
    output = ResidualBlock('Classifier.5', 32, 32, 3, output, resample=None, labels=labels)
    output = ResidualBlock('Classifier.6', 32, 32, 3, output, resample=None, labels=labels)
    # TODO Add normalize
    output = Normalize('Classifier.OutputN', output)
    output = nonlinearity(output)
    output = tf.reduce_mean(output, axis=[2, 3])
    output_cgan = lib.ops.linear.Linear('Classifier.Output', 32, NUM_LABELS, output)
    return output_cgan



def create_model(model_type, model_input_shape, loss_function):
    if model_type.startswith('cnn'):
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(3, 3), padding='same', input_shape=model_input_shape, use_bias=False))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(24, activation='sigmoid'))
        model.compile(loss=loss_function, optimizer=Adam(lr=0.0005), metrics=['accuracy'])
    elif model_type.startswith('rf'):
        regr = RandomForestRegressor(n_estimators=100, max_depth=30, random_state=2)
        return regr
    elif model_type.startswith('svm'):
        regr = SVR(kernel='rbf', C=1e3, gamma=0.1)
        return regr
    elif model_type.startswith('lasso'):
        regr = Lasso()
        return regr
    elif model_type.startswith('lr'):
        regr = LinearRegression()
        return regr
    elif model_type.startswith('ridge'):
        regr = Ridge()
        return regr
    elif model_type.startswith('mlp'):
        regr = MLPRegressor(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(20, 10), random_state=1)
        return regr
    elif model_type.startswith('knn'):
        regr = KNeighborsRegressor()
        return regr
    elif model_type.startswith('elasticnet'):
        regr = ElasticNet(random_state=0)
        return regr
    elif model_type.startswith('extratree'):
        regr = ExtraTreesRegressor(n_estimators=10,
                                   max_features=32,  # Out of 20000
                                   random_state=0)
        return regr
    elif model_type.startswith('dt'):
        regr = DecisionTreeRegressor(max_depth=5)
        return regr
    elif model_type.startswith('gbr'):
        regr = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, max_depth=5))
        return regr
    elif model_type.startswith('ada'):
        regr = MultiOutputRegressor(AdaBoostRegressor(n_estimators=300))
        return regr
    else:
        model = Sequential()
        model.add(Dense(512, activation='relu', input_dim=model_input_shape))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(24, activation='sigmoid'))
        model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])

    return model


## TRAIN
DATAPATH_TRAIN = os.path.join('data', 'train')
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

## VALIDATION
DATAPATH_VALID = './data/valid'
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

## TEST
DATAPATH_TEST = './data/test'
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Select model type.", default="cnn")
    parser.add_argument("-s", "--shape", help="Select input image shape. (rectangle or square?)", default='rect')
    parser.add_argument("-l", "--loss_function", help="Select loss functions.. (rmse,diff_rmse,diff_ce)",
                        default='rmse')
    parser.add_argument("-e", "--epochs", help="Set epochs", default=300)
    parser.add_argument("-b", "--batch_size", help="Set batch size", default=128)
    parser.add_argument("-i", "--experiment_id", help="Select experiment id.", default=1)
    parser.add_argument("-n", "--is_normalized", help="Set is Normalized", action='store_true')
    parser.add_argument("-d", "--data_type", help="Select data type.. (train, valid, test)",
                        default='train')

    args = parser.parse_args()
    model_name = args.model
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    loss_functions = args.loss_function
    input_shape_type = args.shape

    DATAPATH = DATAPATH_TRAIN
    DATASETS = DATASETS_TRAIN

    img_rows, img_cols, channels = 100, 200, 1

    if not input_shape_type.startswith('rect'):
        img_rows, img_cols, channels = 200, 200, 1

    if model_name.startswith('cnn') is False and model_name.startswith('nn') is False:
        img_rows = img_rows // 10
        img_cols = img_cols // 10

    if model_name.startswith('cnn_small'):
        img_rows = img_rows // 5
        img_cols = img_cols // 5

    x_train = []
    y_train = []

    print('Data Loading... Train dataset Start.')

    # load Train dataset
    for data in DATASETS:
        dataframe = pd.read_csv(os.path.join(DATAPATH, '{}.csv'.format(data)), delim_whitespace=False, header=None)
        dataset = dataframe.values

        # split into input (X) and output (Y) variables
        fileNames = dataset[:, 0]
        y_train.extend(dataset[:, 1:25])
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

            if model_name.startswith('cnn') is False and model_name.startswith('nn') is False:
                image = compress_image(image, 10)

            if model_name.startswith('cnn_small'):
                image = compress_image(image, 5)

            if input_shape_type.startswith('rect'):
                x_train.append(image)
            else:
                v_flipped_image = np.flip(image, 0)
                square_image = np.vstack([image, v_flipped_image])
                x_train.append(square_image)

    print('Data Loading... Train dataset Finished.')
    print('Data Loading... Validation dataset Start.')

    DATAPATH = DATAPATH_VALID
    DATASETS = DATASETS_VALID

    x_validation = []
    y_validation = []
    for data in DATASETS:
        dataframe = pd.read_csv(os.path.join(DATAPATH, '{}.csv'.format(data)), delim_whitespace=False, header=None)
        dataset = dataframe.values

        # split into input (X) and output (Y) variables
        fileNames = dataset[:, 0]
        y_validation.extend(dataset[:, 1:25])
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

            if model_name.startswith('cnn') is False and model_name.startswith('nn') is False:
                image = compress_image(image, 10)

            if model_name.startswith('cnn_small'):
                image = compress_image(image, 5)

            if input_shape_type.startswith('rect'):
                x_validation.append(image)
            else:
                v_flipped_image = np.flip(image, 0)
                square_image = np.vstack([image, v_flipped_image])
                x_validation.append(square_image)
    print('Data Loading... Validation dataset Finished.')
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train = np.true_divide(y_train, 2767.1)

    print(x_train.shape, y_train.shape)

    x_validation = np.array(x_validation)
    y_validation = np.array(y_validation)
    y_validation = np.true_divide(y_validation, 2767.1)

    if args.is_normalized:
        print('y_train mean : ', y_train.mean(), np.std(y_train))
        MEAN = 0.5052
        STD = 0.2104
        y_train = scale(y_train, MEAN, STD)
        y_validaton = scale(y_validaton, MEAN, STD)

    if model_name.startswith('cnn'):
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
            y_train = y_train.reshape(y_train.shape[0], channels, img_rows, img_cols)

            x_validation = x_validation.reshape(x_validation.shape[0], channels, img_rows, img_cols)
            y_validaton = y_validaton.reshape(y_validaton.shape[0], channels, img_rows, img_cols)
            input_shape = (channels, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
            x_validation = x_validation.reshape(x_validation.shape[0], img_rows, img_cols, channels)
            input_shape = (img_rows, img_cols, channels)
    else:
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], channels * img_rows * img_cols)
            y_train = y_train.reshape(y_train.shape[0], channels * img_rows * img_cols)
            x_validation = x_validation.reshape(x_validation.shape[0], channels * img_rows * img_cols)
            y_validaton = y_validaton.reshape(y_validaton.shape[0], channels * img_rows * img_cols)
            input_shape = channels * img_rows * img_cols
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols * channels)
            x_validation = x_validation.reshape(x_validation.shape[0], img_rows * img_cols * channels)
            input_shape = channels * img_rows * img_cols

    # for DEBUG
    # print('x shape:', x_train.shape)
    # print('y shape:', y_train.shape)
    # print(x_train.shape[0], 'train samples')

    custom_loss = CustomLoss(loss_functions)
    model = create_model(model_name, input_shape, custom_loss.custom_loss)

    if model_name.startswith('cnn') or model_name.startswith('nn'):
        tic()
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            # pass validtation for monitoring
                            # validation loss and metrics
                            validation_data=(x_validation, y_validation))
        toc()
        score = model.evaluate(x_train, y_train, verbose=0)
        print('Train loss:', score[0])
        print('Train accuracy:', score[1])
        print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

        # serialize model to JSON
        model_json = model.to_json()
        model_export_path_folder = 'models/{}_{}_{}'.format(model_name, batch_size, epochs)
        if not os.path.exists(model_export_path_folder):
            os.makedirs(model_export_path_folder)

        model_export_path_template = '{}/{}_{}_{}.{}'
        model_export_path = model_export_path_template.format(model_export_path_folder, loss_functions,
                                                              input_shape_type, args.experiment_id, 'json')
        with open(model_export_path, "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights(
            model_export_path_template.format(model_export_path_folder, loss_functions, input_shape_type, args.experiment_id, 'h5'))
        print("Saved model to disk")

        # Loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Model - Loss')
        plt.legend(['Training', 'Validation'], loc='upper right')
        train_progress_figure_path_folder = 'result/train_progress'
        if not os.path.exists(train_progress_figure_path_folder):
            os.makedirs(train_progress_figure_path_folder)
        plt.savefig('{}/{}_{}_{}.png'.format(train_progress_figure_path_folder, model_name, loss_functions, args.experiment_id))