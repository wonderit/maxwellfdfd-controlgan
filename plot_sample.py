import pandas as pd
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from keras.models import model_from_json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def get_prediction_model():
    MODEL_JSON_PATH = 'models/cnn_small_rmse_128_300/rmse_rect_1.json'
    MODEL_H5_PATH = 'models/cnn_small_rmse_128_300/rmse_rect_1.h5'

    # load json and create model
    json_file = open(MODEL_JSON_PATH, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(MODEL_H5_PATH)
    return loaded_model


font_size = 20
plt.rcParams.update({'font.size': font_size})

# img_file = '2_1250_8_std0.220.tiff'
# img_file = '2_1250_6_std0.188.tiff'
img_file = '6_1000_9_std0.129.tiff'
img_file = '8_1000_2_std0.125.tiff'
img_file = '7_1000_2_std0.124.tiff'
img_file = '6_1000_3_std0.123.tiff'
img_file = '0_1500_1_std0.241.tiff'
folder_name = './figures/sample/'
csv_path = f'{folder_name}{img_file}_real.csv'
img_path = f'{folder_name}{img_file}'
result_path = f'{folder_name}{img_file}_comparison.png'

img = Image.open(img_path).convert('L')
data = np.array(img, dtype='uint8')
# visually testing our output
data[data > 128] = 255
data[data <= 128] = 0
data_reshaped = np.asarray(data, dtype='uint8')
single_image = data_reshaped // 255
prediction_model = get_prediction_model()
single_image_for_model = single_image.reshape((1, 20, 40, 1))
pred = prediction_model.predict(single_image_for_model)
print('pred', pred.reshape(-1))

x_axis = range(400, 1600, 50)
df = pd.read_csv(csv_path, sep=',', header=None, names=x_axis)

fig, ax = plt.subplots(1, 1, figsize=(14, 7))
ax.plot(x_axis, df.values[0], label='Real', color='black')
myeongjo = 'NanumMyeongjo'
ax.plot(x_axis, pred.reshape(-1), label ='Predict', color = 'blue')

ax.set_title(r'predict simulation', fontsize=font_size, fontname = myeongjo)
ax.set_xlabel('wavelength', fontsize=font_size, fontname = myeongjo)
ax.set_ylabel('transmittance', fontsize=font_size, fontname = myeongjo)
ax.legend(loc='upper left', fontsize=font_size)
ax.grid(True)
ax.set_xlim(400, 1550)
ax.set_ylim(0, 1.01)


# Change major ticks to show every 20.
ax.xaxis.set_major_locator(MultipleLocator(100))
ax.yaxis.set_major_locator(MultipleLocator(0.1))

# Change minor ticks to show every 5. (20/4 = 5)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.grid(which='major', color='#CCCCCC', linestyle='--')
ax.grid(which='minor', color='#CCCCCC', linestyle=':')

fig.tight_layout()
fig.set_size_inches(11, 8)
plt.grid(True)
plt.savefig(result_path, dpi=300)
plt.show()


