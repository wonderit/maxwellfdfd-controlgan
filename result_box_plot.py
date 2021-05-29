from keras.models import model_from_json
import numpy as np
from PIL import Image
import csv
import matplotlib.pyplot as plt
import argparse
import os

# Settings
n_classes = 12
n_count = 10
img_width = 40
img_height = 20
total_width = 480
total_height = 200

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

def get_match_truth_numbers(image_path):
    img = Image.open(image_path).convert('L')
    data = np.array(img, dtype='uint8')
    # visually testing our output
    data[data > 128] = 255
    data[data <= 128] = 0

    data_reshaped = []

    for h in range(total_height // img_height):

        for w in range(total_width // img_width):

            one_img = data[h * img_height:h * img_height + img_height, w * img_width: w * img_width + img_width]
            data_reshaped.append(one_img)

    data_reshaped = np.asarray(data_reshaped)

    prediction_model = get_prediction_model()

    result_match = dict()
    result_truth = dict()
    # Initialize result dict
    for n in range(n_classes):
        wavelength = n * 50 + 1000
        result_match[str(wavelength)] = 0
        result_truth[str(wavelength)] = 0

    for i in range(data_reshaped.shape[0]):
        class_int = (i % n_classes)
        wavelength = class_int * 50 + 1000
        single_image = data_reshaped[i] // 255
        single_image_for_model = single_image.reshape((1, img_height, img_width, 1))
        real = prediction_model.predict(single_image_for_model)

        argsort_top3 = (-real).argsort()[:, :3][0] - 12

        if class_int == argsort_top3[0]:
            result_match[str(wavelength)] += 1

        if argsort_top3[0] > -1:
            result_truth[str(argsort_top3[0]* 50 + 1000)] += 1

    return result_match, result_truth

def print_csv(folder_path, result_m, result_t):
    result_folder = f'{folder_path}/csv'

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    csv_file_result_match = "{}/result_match.csv".format(result_folder)
    csv_file_result_truth = "{}/result_truth.csv".format(result_folder)

    keys = result_m[0].keys()

    with open(csv_file_result_match, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(result_m)

    with open(csv_file_result_truth, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(result_t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-sns", "--sample_number_start", help="Select sample_number", type=int, default=35099)
    parser.add_argument("-sne", "--sample_number_end", help="Select sample_number", type=int, default=36000)
    parser.add_argument("-fn", "--folder_name", help="Select folder name", default='./logs/wgan_new')

    args = parser.parse_args()
    folder_name = args.folder_name

    # result_match_all = dict()
    # result_truth_all = dict()
    result_match_list = list()
    result_truth_list = list()

    # Initialize result dict
    for n in range(n_classes):
        wavelength = n * 50 + 1000
        # result_match_all[str(wavelength)] = [0 for x in range(n_count)]
        # result_truth_all[str(wavelength)] = [0 for x in range(n_count)]

    for image_number in range(args.sample_number_start, args.sample_number_end, 100):
        file_name = f'samples_{image_number}'
        img_path = f'{folder_name}/{file_name}.png'
        rm, rt = get_match_truth_numbers(img_path)
        result_match_list.append(rm)
        result_truth_list.append(rt)

    print_csv(folder_name, result_match_list, result_truth_list)






