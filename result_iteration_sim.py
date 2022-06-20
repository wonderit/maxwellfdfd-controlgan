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

def get_accuracy(image_path, save_csv=False, folder_path='.'):
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

    # print('match_cnt : {}, \t correct_cnt_top3 : {}, \t correct_cnt_top5 : {}'.format(sum(result_match.values()), sum(result_top3.values()), sum(result_top5.values())))

    percent_match = sum(result_match.values()) / (n_classes * n_count)
    percent_top3 = sum(result_top3.values()) / (n_classes * n_count)
    percent_top5 = sum(result_top5.values()) / (n_classes * n_count)

    print('percent_match : {0:.4f} \t top3 : {1:.4f} \t top5 : {2:.4f}'.format(percent_match, percent_top3, percent_top5))
    if save_csv:
        result_folder = f'{folder_path}/example_result/{file_name}'
        result_png = f'{folder_path}/example_result/{file_name}/new_{file_name}.png'

        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        plt.imsave(result_png, data, cmap='Greys')

        csv_file_result_match = "{}/{}_result_match.csv".format(result_folder, file_name)
        csv_file_result_top3 = "{}/{}_result_top3.csv".format(result_folder, file_name)
        csv_file_result_top5 = "{}/{}_result_top5.csv".format(result_folder, file_name)
        csv_file_result_truth = "{}/{}_result_truth.csv".format(result_folder, file_name)
        text_file_result = "{}/{}_result.txt".format(result_folder, file_name)

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
        with open(text_file_result, "w+") as file:
            file.write('match_cnt : {}, \t correct_cnt_top3 : {}, \t correct_cnt_top5 : {}\n'.format(
                    sum(result_match.values()), sum(result_top3.values()), sum(result_top5.values())))
            file.write('percent_match : {0:.4f} \t top3 : {1:.4f} \t top5 : {2:.4f}'.format(percent_match, percent_top3,
                                                                                           percent_top5))

    return percent_match, percent_top3, percent_top5


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-ns", "--sample_number_start", help="Select sample_number", type=int, default=1)
    parser.add_argument("-ne", "--sample_number_end", help="Select sample_number", type=int, default=10)

    args = parser.parse_args()
    folder_name = './generated-lr2e4-lambda1'

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
    # epochs = 10
    for image_number in range(args.sample_number_start, args.sample_number_end):
        file_name = 'generated-images-{0:0=4d}'.format(image_number)
        # file_name = f'generated-images-{image_number}.png'
        img_path = f'{folder_name}/{file_name}.png'
        top_acc_i, top3_acc_i, top5_acc_i = get_accuracy(img_path, True, folder_name)
        if top_accuracy < top_acc_i:
            top_accuracy = top_acc_i
            top3_accuracy = top3_acc_i
            top5_accuracy = top5_acc_i
            top_accuracy_id = img_path

    print(f'Top Accuracy : {top_accuracy}, top3: {top3_accuracy}, top5: {top5_accuracy}, sample number:{top_accuracy_id}')
    # get_accuracy(top_accuracy_id, True, folder_name)






