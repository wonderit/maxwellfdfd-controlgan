from keras.models import model_from_json
import numpy as np
from PIL import Image
import csv
import matplotlib.pyplot as plt
import argparse
import os

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--sample_number", help="Select sample_number", default="29999")

    args = parser.parse_args()
    file_name = f'samples_{args.sample_number}'
    result_folder = 'logs/cgan/{}'.format(file_name)
    result_png = 'logs/cgan/{}/new_{}.png'.format(file_name, file_name)

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    img_path = 'logs/cgan/{}.png'.format(file_name)
    img = Image.open(img_path).convert('L')
    data = np.array(img, dtype='uint8')

    # visually testing our output
    data[data>128] = 255
    data[data<=128] = 0
    plt.imsave(result_png, data, cmap='Greys')
    # plt.figure()
    # plt.imshow(data, cmap='Greys')
    # plt.show()

    #
    data_reshaped = []
    n_classes = 12
    n_count = 10
    img_width = 40
    img_height = 20

    total_width = 480
    total_height = 200

    for h in range(total_height // img_height):

        for w in range(total_width // img_width):

            one_img = data[h * img_height:h * img_height + img_height, w * img_width: w * img_width + img_width]
            data_reshaped.append(one_img)

    data_reshaped = np.asarray(data_reshaped)

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
        wavelength = class_int * 50 + 1000
        single_image = data_reshaped[i] // 255
        single_image_for_model = single_image.reshape((1, img_height, img_width, 1))
        real = prediction_model.predict(single_image_for_model)

        argsort_top5 = (-real).argsort()[:, :5][0] - 12
        argsort_top3 = (-real).argsort()[:, :3][0] - 12

        if class_int in argsort_top5:
            result_top5[str(wavelength)] += 1

        if class_int in argsort_top3:
            result_top3[str(wavelength)] += 1

        if class_int == argsort_top3[0]:
            result_match[str(wavelength)] += 1

        if argsort_top3[0] > -1:
            result_truth[str(argsort_top3[0]* 50 + 1000)] += 1

    print('match_cnt : {}, \t correct_cnt_top3 : {}, \t correct_cnt_top5 : {}'.format(sum(result_match.values()), sum(result_top3.values()), sum(result_top5.values())))

    percent_match = sum(result_match.values()) / (n_classes * n_count)
    percent_top3 = sum(result_top3.values()) / (n_classes * n_count)
    percent_top5 = sum(result_top5.values()) / (n_classes * n_count)

    print('percent_match : {0:.4f} \t top3 : {1:.4f} \t top5 : {2:.4f}'.format(percent_match, percent_top3, percent_top5))

    csv_file_result_match = "{}/{}_result_match.csv".format(result_folder, file_name)
    csv_file_result_top3 = "{}/{}_result_top3.csv".format(result_folder, file_name)
    csv_file_result_top5 = "{}/{}_result_top5.csv".format(result_folder, file_name)
    csv_file_result_truth = "{}/{}_result_truth.csv".format(result_folder, file_name)

    a_file = open(csv_file_result_match, "w")

    writer = csv.writer(a_file)
    for key, value in result_match.items():
        writer.writerow([key, value])

    a_file.close()


    a_file = open(csv_file_result_top3, "w")

    writer = csv.writer(a_file)
    for key, value in result_top3.items():
        writer.writerow([key, value])

    a_file.close()


    a_file = open(csv_file_result_truth, "w")

    writer = csv.writer(a_file)
    for key, value in result_truth.items():
        writer.writerow([key, value])

    a_file.close()


    a_file = open(csv_file_result_top5, "w")

    writer = csv.writer(a_file)
    for key, value in result_top5.items():
        writer.writerow([key, value])

    a_file.close()



