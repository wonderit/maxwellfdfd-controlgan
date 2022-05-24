import pandas as pd
from keras import backend as K
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from PIL import Image


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        toc_ = "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
        print(toc_)
        return toc_
    else:
        toc_ = "Toc: start time not set"
        print(toc_)
        return toc_

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
DATAPATH = './data/test'

model_name_details = [
    'cnn_small_rmse_128_300/rmse_rect_1',
    # 'cnn_rmse_diff_ce_128_300/rmse,diff_ce_rect_1',
    # 'cnn_rmse_diff_rmse_128_300/rmse,diff_rmse_rect_1',
]

label_name = [
    'cnn_small1',
    'cnn_small2',
    'cnn_small3'
]


colors=[
    'green', 'skyblue', 'red'
]

if MODEL_SHAPE_TYPE == 'rect':
    img_rows, img_cols, channels = 100, 200, 1
else:
    img_rows, img_cols, channels = 200, 200, 1

img_rows = img_rows // 5
img_cols = img_cols // 5


RESULT_PATH = './result/predict'
model_name = './models'

x_test = []
y_test = []


# data_folder = 'binary_test_1001'
# data_id = 63409

# data_folder = 'binary_rl_fix_1014'
# data_id = 756 # 852

# data_folder = 'binary_new_test_501'
# data_id = 131  # 282, 168 132
#
# # #
# #
# # data_folder = 'binary_rl_fix_test_1003'
# # data_id = 557  # 374, 137 557
#
# # data_folder = 'binary_rl_fix_test_1002'
# # data_id = 966
#
# data_folder = 'binary_rl_fix_test_1003'
# data_id = 374

# data_folder = 'binary_rl_limit_test_1001'
# data_id = 9
#
# data_folder = 'binary_rl_test_501'
# data_id = 9

# rank 1
# data_folder = 'binary_new_test_1501'
# data_id = 783   # 259 617, 783
# #
# data_folder = 'binary_rl_fix_test_1005'
# data_id = 205   # 259 614 1339
# 1~ 500
# data_folder = 'binary_new_test_501'
# data_id = 125  # 282, 168 132 293 181 121 178 498

# 501 ~ 2000
data_folder = 'binary_new_test_1501'
data_id = 979   # 259 614 1339 1459 886 927 1481 890 751 785 1208

# 2000 ~ 2999
# data_folder = 'binary_rl_fix_1014'
# data_id = 470  # 724, 793 113

# 3000 ~ 3999
# data_folder = 'binary_rl_fix_1015'
# data_id = 496  # 935,206
# 4000 ~ 4999
data_folder = 'binary_rl_fix_test_1002'
data_id = 425  # 870 959

# 5000 ~ 5999
# data_folder = 'binary_rl_fix_test_1003'
# data_id = 808  # 557 808 178 903


# 6000 ~ 6999
# data_folder = 'binary_rl_fix_test_1004'
# data_id = 146  # 792 146


# 7000 ~ 7999
# data_folder = 'binary_rl_fix_test_1005'
# data_id = 959  # 870 959

#  8000 ~
# data_folder = 'binary_test_1101'
# data_id = 245   # 288, 514, 928, 35, 220 329 930 493 167 245 632 517 985

# image = cv2.imread('{}/{}/{}.tiff'.format(DATAPATH, data_folder, data_id), 0)
# image = np.array(image, dtype=np.uint8)
# image //= 255

# load image

image = Image.open('{}/{}/{}.tiff'.format(DATAPATH, data_folder, data_id))
image = np.array(image, dtype=np.uint8)
image = compress_image(image, 5)


if MODEL_SHAPE_TYPE == 'rect':
    x_test.append(image)
else:
    v_flipped_image = np.flip(image, 0)
    square_image = np.vstack([image, v_flipped_image])
    x_test.append(square_image)

dataframe = pd.read_csv('{}/{}.csv'.format(DATAPATH, data_folder), delim_whitespace=False, header=None)
dataset = dataframe.values
fileNames = dataset[:, 0]

for idx, val in enumerate(fileNames):
    if int(idx+1) == int(data_id):
        y_test = dataset[idx, 1:25]
        break

x_test = np.array(x_test)
y_test = np.array(y_test)
y_test = np.true_divide(y_test, 2767.1)

if K.image_data_format() == 'channels_first':
    x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
    y_test = y_test.reshape(y_test.shape[0], channels, img_rows, img_cols)
    input_shape = (channels, img_rows, img_cols)
else:
    x_test = x_test.reshape(1, img_rows, img_cols, channels)
    input_shape = (img_rows, img_cols, channels)

result = dict()
result['real'] = x_test
x_axis = range(400, 1600, 50)
fig, ax = plt.subplots(1, 1, figsize=(14, 7))
ax.plot(x_axis, y_test, label='real', color='black')
MODEL_JSON_PATH = ''
MODEL_H5_PATH = ''
myeongjo = 'NanumMyeongjo'

for i, model_name_detail in enumerate(model_name_details):
    MODEL_JSON_PATH = '{}/{}.json'.format(model_name, model_name_detail)
    MODEL_H5_PATH = '{}/{}.h5'.format(model_name, model_name_detail)
    print("Loaded model : {}".format(model_name_detail))

    # load json and create model
    json_file = open(MODEL_JSON_PATH, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(MODEL_H5_PATH)

    # evaluate loaded model on test data
    loaded_model.compile(loss=root_mean_squared_error, optimizer='adam', metrics=['accuracy'])

    tic()
    if model_name_detail.startswith('cnn'):
        y_predict = loaded_model.predict(x_test)
    else:
        x_test_nn = x_test.reshape(x_test.shape[0], img_rows * img_cols * channels)
        y_predict = loaded_model.predict(x_test_nn)
    toc()

    ax.plot(x_axis, y_predict[0], label = label_name[i], color = colors[i])

    # plot model
    # plot_model(loaded_model, to_file='plot_model_{}_{}.png'.format(data_folder, data_id), show_shapes=True, show_layer_names=True)

peaks_positive, _ = find_peaks(y_test, height=0)
peaks_negative, _ = find_peaks(1 - y_test, height=0)
mask = np.zeros_like(y_test, np.bool)
mask[peaks_positive] = 1
mask[peaks_negative] = 1
peak_array = mask * y_test
peak_array[peak_array == 0] = np.nan
plt.plot(x_axis, peak_array, "o", markersize=10)

ax.set_title(r'predict simulation', fontsize = 14, fontname = myeongjo)
ax.set_xlabel('wavelength', fontsize = 14, fontname = myeongjo)
ax.set_ylabel('transmittance', fontsize = 14, fontname = myeongjo)
ax.legend(loc = 'upper left', fontsize = 14)
# ax.legend(loc = 'lower center', fontsize = 14)
ax.grid(True)

# ax.set_ylim(0, 10000)
# ax.set_yticks(np.arange(0, 10000 + 1, 2500))

fig.tight_layout()
fig.set_size_inches(11,8)
plt.savefig('plt_rmse_type1_2_all_{}_{}.png'.format(data_folder, data_id))
plt.show()