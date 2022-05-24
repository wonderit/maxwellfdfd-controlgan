from math import sqrt
from PIL import Image
import pandas as pd
from keras import backend as K
from keras import losses
from keras.layers import Average
from keras.models import Model
from keras.models import model_from_json
from sklearn.metrics import mean_squared_error, r2_score
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class CustomLoss():
    def __init__(self, loss_functions):
        super(CustomLoss, self).__init__()
        self.loss_function_array = loss_functions.split(',')
        print(self.loss_function_array)

    def tf_diff_axis_1(self, a):
        return a[:, 1:] - a[:, :-1]

    def custom_loss(self, y_true, y_pred):
        loss = 0
        y_true_diff = self.tf_diff_axis_1(y_true)
        y_pred_diff = self.tf_diff_axis_1(y_pred)
        threshold_value = 0
        y_true_diff_binary = K.cast(K.greater(y_true_diff, threshold_value), K.floatx())
        y_pred_diff_binary = K.cast(K.greater(y_pred_diff, threshold_value), K.floatx())
        if 'rmse' in self.loss_function_array:
            loss = loss + K.sqrt(K.mean(K.square(y_pred - y_true)))

        if 'diff_rmse' in self.loss_function_array:
            loss = loss + K.sqrt(K.mean(K.square(y_pred_diff - y_true_diff)))

        if 'diff_ce' in self.loss_function_array:
            loss = loss + losses.binary_crossentropy(y_true_diff, y_pred_diff)

        if 'diff_bce' in self.loss_function_array:
            loss = loss + losses.binary_crossentropy(y_true_diff_binary, y_pred_diff_binary)

        return loss

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def normalized_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square((y_pred - y_true) / 2600 / (y_true / 2600)), axis=-1))

def ensemble(models, model_input):
    outputs = [model(model_input) for model in models]
    y = Average()(outputs)
    model = Model(model_input, y, name='ensemble')
    return model

def ensembleModels(models, model_input):
    # collect outputs of models in a list
    yModels = [model(model_input) for model in models]
    # averaging outputs
    yAvg = Average(yModels)
    # build model from same input and avg output
    modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')

    return modelEns

def tf_diff(a):
    return a[1:] - a[:-1]


def tf_diff_axis_1(a):
    return a[:, 1:] - a[:, :-1]

def image_trim(image, x=8, y=8):
    print(image.shape)
    images = []
    width = image.shape[1] // x
    height = image.shape[0] // y
    print(width, height)
    for i in range(0, x):
        for j in range(0, y):
            trimmed_image = image[j*height:j*height+height, i*width:i*width + width]
            resized_image = cv2.resize(trimmed_image, None, fx=5, fy=5, interpolation=cv2.INTER_AREA)
            cv2.imwrite('./data_test/image_from_gan/' + str(j) + '_' + str(i) + '.tiff', resized_image)
            resized_image //= 255
            images.append(resized_image)
    return images

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

# PARAMETERS
MODEL_SHAPE_TYPE = 'rect'

## TEST
DATAPATH = os.path.join('data', 'test')
DATASETS = [
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

# Test Train
DATAPATH = DATAPATH_TRAIN
DATASETS = DATASETS_TRAIN


model_names = [
    'RMSE',
    'RMSE',
    'RMSE',
]

model_name_details = [
    'cnn_small_128_300/rmse_rect_1',
    'cnn_small_128_300/rmse_rect_2',
    'cnn_small_128_300/rmse_rect_3',
]

colors=[
    'green', 'skyblue', 'red',
]

model_folder_path = 'models'
is_mean_std = False

if MODEL_SHAPE_TYPE == 'rect':
    img_rows, img_cols, channels = 100, 200, 1
else:
    img_rows, img_cols, channels = 200, 200, 1


img_rows_compressed = img_rows // 10
img_cols_compressed = img_cols // 10

#     For cnn small
img_rows = img_rows // 5
img_cols = img_cols // 5

lowest_RMSE = 999
lowest_RMSE_id = 0

lowest_RMSE_DIFF_RMSE = 999
lowset_RMSE_DIFF_RMSE_ID = 0

lowset_local_RMSE = 999
lowset_local_RMSE_id = 0

lowest_POLY_RMSE = 999
lowest_POLY_RMSE_ID = 0
model_name = './'

x_test = []
x_test_compressed = []
y_test = []
y_test_compressed = []

print('Data Loading....')

# load dataset
for i, data in enumerate(DATASETS):

    dataframe = pd.read_csv('{}/{}.csv'.format(DATAPATH, data), delim_whitespace=False, header=None)
    dataset = dataframe.values
    # split into input (X) and output (Y) variables
    fileNames = dataset[:, 0]
    y_test.extend(dataset[:, 1:25])
    for idx, file in enumerate(fileNames):

        try:
            image = Image.open(os.path.join(DATAPATH, data, '{}.tiff'.format(int(file))))
            image = np.array(image, dtype=np.uint8)
        except (TypeError, FileNotFoundError) as te:

            image = Image.open(os.path.join(DATAPATH, data, '{}.tiff'.format(idx + 1)))
            # image = cv2.imread('{}/{}/{}.tiff'.format(DATAPATH, data, idx + 1), 0)

            image = np.array(image, dtype=np.uint8)

        compressed_image = compress_image(image, 10)
        image = compress_image(image, 5)

        if MODEL_SHAPE_TYPE.startswith('rect'):
            x_test.append(image)
            x_test_compressed.append(compressed_image)
        else:
            v_flipped_image = np.flip(image, 0)
            square_image = np.vstack([image, v_flipped_image])
            x_test.append(square_image)

            v_flipped_image_compressed = np.flip(compressed_image, 0)
            square_image_compressed = np.vstack([compressed_image, v_flipped_image_compressed])
            x_test_compressed.append(square_image_compressed)


print(f'Data Loading... Finished. row,col=({img_rows}, {img_cols})')

x_test = np.array(x_test)
x_test_compressed = np.array(x_test_compressed)
y_test = np.array(y_test)
y_test = np.true_divide(y_test, 2767.1)

if K.image_data_format() == 'channels_first':
    x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
    y_test = y_test.reshape(y_test.shape[0], channels, img_rows, img_cols)
    x_test_compressed = x_test_compressed.reshape(x_test.shape[0], channels * img_rows_compressed * img_cols_compressed)
    # y_test_compressed = y_test.reshape(y_test.shape[0], channels * img_rows_compressed * img_cols_compressed)
    input_shape = (channels, img_rows, img_cols)
    input_shape_compressed = channels*img_rows_compressed*img_cols_compressed
else:
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
    x_test_compressed = x_test_compressed.reshape(x_test_compressed.shape[0], channels*img_rows_compressed*img_cols_compressed)
    # y_test_compressed = y_test.reshape(y_test.shape[0], channels*img_rows_compressed*img_cols_compressed)
    input_shape = (img_rows, img_cols, channels)
    input_shape_compressed = channels * img_rows_compressed * img_cols_compressed

result = dict()
result['real'] = x_test
x_axis = range(400, 1600, 50)
# fig, ax = plt.subplots(1, 1, figsize=(14, 7))
# ax.plot(x_axis, y_test, label='real', color='black')
MODEL_JSON_PATH = ''
MODEL_H5_PATH = ''
myeongjo = 'NanumMyeongjo'

mask_array = np.ones_like(y_test, np.bool)

for j in range(len(y_test)):
    peaks_positive, _ = find_peaks(y_test[j], height=0)
    peaks_negative, _ = find_peaks(1 - y_test[j], height=0)
    mask = np.ones(len(y_test[j]), np.bool)
    mask[peaks_positive] = 0
    mask[peaks_negative] = 0
    mask_array[j][mask] = 0

result_runningTime = dict()
result_r2 = dict()
result_r2_local_minmax = dict()
result_rmse = dict()
result_rmse2 = dict()
result_diff_rmse = dict()
result_rmse_add_diff_rmse = dict()
result_poly = dict()
rmse_for_boxplot = dict()
rmse_local_for_boxplot = dict()


result_list = []
for i, model_name_detail in enumerate(model_name_details):
    print(model_name_detail)
    parsed_model_name = model_name_detail.split('/')[0]
    runningTime = 0
    if model_name_detail.startswith('cnn') or model_name_detail.startswith('nn'):
        parsed_model_name = model_name_detail.split('/')[0] + '_' + model_name_detail.split('/')[1]
        MODEL_JSON_PATH = '{}/{}.json'.format(model_folder_path, model_name_detail)
        MODEL_H5_PATH = '{}/{}.h5'.format(model_folder_path, model_name_detail)
        # load json and create model
        json_file = open(MODEL_JSON_PATH, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(MODEL_H5_PATH)
        print("Loaded model from disk")

        if model_name_detail.startswith('cnn'):
            tic()
            y_predict = loaded_model.predict(x_test)
        else:
            x_test_nn = x_test.reshape(x_test.shape[0], img_rows * img_cols * channels)
            tic()
            y_predict = loaded_model.predict(x_test_nn)
        runningTime = toc()

    if is_mean_std == True:
        MEAN = 0.5052
        STD = 0.2104
        y_predict = rescale(y_predict, MEAN, STD)


    # corr = np.corrcoef(y_test, y_predict)[0, 1]
    r2 = r2_score(y_test, y_predict)

    meanSquaredError = mean_squared_error(y_test, y_predict)
    rmse = sqrt(meanSquaredError)

    rmse_all = []
    count = 0

    message = 'r2:{0:.4f}, RMSE:{1:.4f}'.format(r2, rmse)

    rmse_for_boxplot[parsed_model_name] = rmse_all
    y_test_for_local_minmax = y_test[mask_array]
    y_predict_for_local_minmax = y_predict[mask_array]

    y_test_for_local_minmax_inverse = y_test[~mask_array]
    y_predict_for_local_minmax_inverse = y_predict[~mask_array]

    rmse2 = sqrt(mean_squared_error(y_test_for_local_minmax, y_predict_for_local_minmax))
    r2_local_minmax = r2_score(y_test_for_local_minmax, y_predict_for_local_minmax)
    result_rmse2[parsed_model_name] = rmse2
    result_r2[parsed_model_name] = r2
    result_r2_local_minmax[parsed_model_name] = r2_local_minmax
    result_rmse[parsed_model_name] = rmse
    result_runningTime[parsed_model_name] = runningTime

    y_test_diff = tf_diff_axis_1(y_test)
    y_predict_diff = tf_diff_axis_1(y_predict)
    mse_diff = mean_squared_error(y_test_diff, y_predict_diff)
    rmse_diff = sqrt(mse_diff)
    result_diff_rmse[parsed_model_name] = rmse_diff

    result_rmse_add_diff_rmse[parsed_model_name] = rmse_diff + rmse

    plt.scatter(y_predict_for_local_minmax_inverse, y_test_for_local_minmax_inverse, s=3, alpha=0.3, label='all', marker='+')
    plt.scatter(y_predict_for_local_minmax, y_test_for_local_minmax, s=2, alpha=0.3, label='local_minmax', marker='.')
    # x_margin = -0.05
    x_margin = 0
    plt.text(x_margin, 1, 'R² = %0.4f' % r2)
    plt.text(x_margin, 0.95, 'RMSE = %0.4f' % rmse)
    plt.text(x_margin, 0.9, 'local minmax R² = %0.4f' % r2_local_minmax)
    plt.text(x_margin, 0.85, 'local minmax RMSE = %0.4f' % rmse2)
    plt.xlabel('Predictions')
    plt.ylabel('Actual')
    plt.savefig("{}/scatter_alpha/{}_all.png".format('result', parsed_model_name))
    plt.clf()

print('running time:', result_runningTime)
print('rmse: ', result_rmse)
print('rmse local minmax: ', result_rmse2)
print('r2: ', result_r2)
print('r2-local: ', result_r2_local_minmax)