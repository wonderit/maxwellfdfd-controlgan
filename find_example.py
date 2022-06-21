from keras.models import model_from_json
from PIL import Image
import matplotlib.pyplot as plt
import os
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


folder_name = './figures/sample'

n_classes = 12
n_count = 10
img_width = 40
img_height = 20

total_width = 480
total_height = 200

top_accuracy = 0.0
top3_accuracy = 0.0
top5_accuracy = 0.0
top_accuracy_id = 0

results = []
sample_number_start = 0
sample_number_end = 9
target_class = 10 # 0, 5, 10

result_folder = f'{folder_name}/class_{target_class * 50 + 1000}'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

for image_number in range(sample_number_start, sample_number_end):

    file_name = f'controlgan_{image_number}'

    img_path = f'{folder_name}/{file_name}.png'

    img = Image.open(img_path).convert('L')
    data = np.array(img, dtype='uint8')
    # visually testing our output
    data[data > 128] = 255
    data[data <= 128] = 0
    # data[data > 128] = 0
    # data[data <= 128] = 255

    data_reshaped = []

    for h in range(total_height // img_height):

        for w in range(total_width // img_width):

            one_img = data[h * img_height:h * img_height + img_height, w * img_width: w * img_width + img_width]
            data_reshaped.append(one_img)

    data_reshaped = np.asarray(data_reshaped, dtype='uint8')

    prediction_model = get_prediction_model()

    match_cnt = 0
    correct_cnt_top3 = 0
    correct_cnt_top5 = 0
    result_match = dict()
    result_top3 = dict()
    result_top5 = dict()
    result_truth = dict()
    # Initialize result dict
    for n in range(n_classes):
        wavelength = n * 50 + 1000
        result_match[str(wavelength)] = 0
        result_top3[str(wavelength)] = 0
        result_top5[str(wavelength)] = 0
        result_truth[str(wavelength)] = 0

    print('# of samples', data_reshaped.shape[0])

    for i in range(data_reshaped.shape[0]):
        class_int = (i % n_classes)

        # pass class
        if not class_int == target_class:
            continue

        wavelength = class_int * 50 + 1000
        single_image = data_reshaped[i] // 255
        reverse_single_image = ~(data_reshaped[i] // 255)
        single_image_for_model = single_image.reshape((1, img_height, img_width, 1))
        real = prediction_model.predict(single_image_for_model)
        std_real = np.std(real)

        argsort_top5 = (-real).argsort()[:, :5][0] - 12
        argsort_top3 = (-real).argsort()[:, :3][0] - 12

        argsort_top5[argsort_top5 < 0] = 0
        argsort_top3[argsort_top3 < 0] = 0
        label = argsort_top3[0]

        if not class_int == argsort_top3[0]:
            continue
        result_png = f'{result_folder}/{image_number}_{wavelength}_{i // n_classes}_std{std_real:.3f}.tiff'
        plt.imsave(result_png, reverse_single_image, cmap='Greys')
