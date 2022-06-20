import numpy as np
import pandas as pd
import os
import imageio
from PIL import Image
from typing import Any, Tuple
from torch.utils.data import DataLoader

import torch
import torch.utils.data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.utils import save_image
from tqdm import tqdm
import torch.nn.functional as F
from keras.models import model_from_json
import csv
import argparse

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

NUM_LABELS = 12
NUM_SAMPLES_PER_LABEL = 10
latent_size = 116
IMAGE_WIDTH = 40
IMAGE_HEIGHT = 20
clamp_num=0.01  # WGAN clip gradient

def compress_image(prev_image, n):
    if n < 2:
        return prev_image

    height = prev_image.shape[0] // n
    width = prev_image.shape[1] // n
    new_image = np.zeros((height, width), dtype="uint8")
    for i in range(0, height):
        for j in range(0, width):
            new_image[i, j] = prev_image[n * i, n * j]

    return new_image

class CEMDataset(torch.utils.data.Dataset):
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

    def __init__(self,
                 root: str,
                 train: bool = True,
                 scale: int = 1,
                 is_regression: bool = True,
                 ) -> None:
        self.train = train
        self.root = root
        self.scale = scale
        self.width = 200 // scale
        self.height = 100 // scale
        self.is_regression = is_regression

        if self.train:
            DATAPATH = os.path.join(root, 'train')
            DATASETS = self.DATASETS_TRAIN
        else:
            DATAPATH = os.path.join(root, 'test')
            DATASETS = self.DATASETS_TEST

        self.data: Any = []
        self.targets = []

        print('data loading ... ')

        # load Train dataset
        for data in DATASETS:
            dataframe = pd.read_csv(os.path.join(DATAPATH, '{}.csv'.format(data)), delim_whitespace=False, header=None)
            dataset = dataframe.values

            # split into input (X) and output (Y) variables
            fileNames = dataset[:, 0]

            # 1. first try max
            dataset[:, 1:25] /= 2767.1

            # 2. Classification or Regression
            labels = dataset[:, 13:25]
            if self.is_regression:
                self.targets.extend(labels)
            else:
                labels = np.apply_along_axis(lambda x: np.argmax(x), 1, labels)
                self.targets.extend(labels)

            for idx, file in enumerate(fileNames):
                try:
                    image = Image.open(os.path.join(DATAPATH, data, '{}.tiff'.format(int(file))))
                    image = np.array(image).astype(np.uint8)
                except (TypeError, FileNotFoundError) as te:
                    image = Image.open(os.path.join(DATAPATH, data, '{}.tiff'.format(idx + 1)))
                    try:
                        image = np.array(image).astype(np.uint8)
                    except:
                        continue
                image = compress_image(image, self.scale)
                self.data.append(np.array(image).flatten(order='C'))

        self.data = np.vstack(self.data).reshape(-1, 1, self.height, self.width)
        self.data = self.data.transpose((0, 1, 2, 3))  # convert to HWC CHW
        print(f'Data Loading Finished. len : {len(self.data)}')

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target

    def __len__(self) -> int:
        return len(self.data)

def denorm(img_tensors):
    return img_tensors * 1.

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=False)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


class CnnModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            # in: 1 x 20 x 40
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=(0, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=(3, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=(3, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # out: 1 x 1 x 1
            nn.Flatten(),
            nn.Linear(3840, 64),
            nn.Linear(64, 12),
            nn.Sigmoid()
    )

    def forward(self, xb):
        return self.network(xb)

def weight_init(m):
    # weight_initialization: important for wgan
    class_name=m.__class__.__name__
    if class_name.find('Conv')!=-1:
        m.weight.data.normal_(0,0.02)
    elif class_name.find('Norm')!=-1:
        m.weight.data.normal_(1.0,0.02)


def train_discriminator(real_images, real_labels, opt_d):
    # Clear discriminator gradients
    opt_d.zero_grad()

    # Pass real images through discriminator
    real_preds = discriminator(real_images.float())

    real_preds = real_preds.view(-1, 1)
    real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()

    # Generate fake images by regression label
    label_encoded = real_labels.float()
    z = torch.randn(real_images.size(0), latent_size).to(device)
    z_concat = torch.cat((z, label_encoded), 1)
    z_concat = z_concat[:, :, None, None]
    fake_images = generator(z_concat)

    # Pass fake images through discriminator
    fake_preds = discriminator(fake_images)
    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights (DCGAN)
    # loss = real_loss + fake_loss
    # Wgan Loss
    loss = - (real_preds.mean() - fake_preds.mean())
    loss.backward()
    opt_d.step()
    return loss.item(), real_loss.item(), fake_loss.item()


def train_generator(real_labels, opt_g):
    # Clear generator gradients
    opt_g.zero_grad()

    # Generate fake images by regression label
    label_encoded = real_labels.float()
    z = torch.randn(real_labels.size(0), latent_size).to(device)
    z_concat = torch.cat((z, label_encoded), 1)
    z_concat = z_concat[:, :, None, None]
    fake_images = generator(z_concat)

    # Try to fool the discriminator
    preds = discriminator(fake_images)
    targets = torch.ones(real_labels.size(0), 1, device=device)
    wgan_loss = -torch.mean(preds)

    # add simulator loss
    simulator.eval()
    simulation = simulator(fake_images)
    simulation_loss = F.mse_loss(simulation, label_encoded.float())
    # loss = 0.1 * bce_loss + c_loss

    # WGAN
    # loss = wgan_loss

    # WGAN + Simulation Loss (Lambda = 0.1)
    loss = LAMBDA * wgan_loss + simulation_loss

    # Update generator weights
    loss.backward()
    opt_g.step()

    return loss.item(), simulation_loss



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


def eval_samples(index, fake_images ):

    result_match = dict()
    result_top3 = dict()
    result_top5 = dict()
    result_truth = dict()
    # Initialize result dict
    for n in range(NUM_LABELS):
        wavelength = n * 50 + 1000
        result_match[str(wavelength)] = 0
        result_top3[str(wavelength)] = 0
        result_top5[str(wavelength)] = 0
        result_truth[str(wavelength)] = 0

    np_fake_images = fake_images.cpu().detach().numpy()
    np_fake_images[np_fake_images > 0.5] = 1.0
    np_fake_images[np_fake_images <= 0.5] = 0.0
    prediction_model = get_prediction_model()
    for i in range(np_fake_images.shape[0]):
        np_fake_image = np_fake_images[i]
        np_fake_image = np_fake_image.reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, 1))
        real = prediction_model.predict(np_fake_image)
        class_int = (i % NUM_LABELS)

        argsort_top5 = (-real).argsort()[:, :5][0] - 12
        argsort_top3 = (-real).argsort()[:, :3][0] - 12

        if class_int in argsort_top5:
            result_top5[str(wavelength)] += 1

        if class_int in argsort_top3:
            result_top3[str(wavelength)] += 1

        if class_int == argsort_top3[0]:
            result_match[str(wavelength)] += 1

        if argsort_top3[0] > -1:
            result_truth[str(argsort_top3[0] * 50 + 1000)] += 1

        percent_match = sum(result_match.values()) / (NUM_LABELS * NUM_SAMPLES_PER_LABEL)
        percent_top3 = sum(result_top3.values()) / (NUM_LABELS * NUM_SAMPLES_PER_LABEL)
        percent_top5 = sum(result_top5.values()) / (NUM_LABELS * NUM_SAMPLES_PER_LABEL)

    print(
        'percent_match : {0:.4f} \t top3 : {1:.4f} \t top5 : {2:.4f}'.format(percent_match, percent_top3, percent_top5))

    file_name = f'samples_{index}'
    result_folder = f'{sample_dir}/example_result/{file_name}'
    result_png = f'{sample_dir}/example_result/{file_name}/new_{file_name}.png'

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    plt.imsave(result_png, np_fake_images, cmap='Greys')

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


def save_samples(index, latent_tensors):
    fake_images = generator(latent_tensors)
    # eval_samples(index, fake_images)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), show=False, nrow=12)
    print('Saving', fake_fname)


def fit(epochs, lr, start_idx=1):
    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    losses_c = []

    # Create optimizers
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for real_images, real_labels in tqdm(train_dl):
            # Train discriminator
            # modification: clip param for discriminator (WGAN)
            for parm in discriminator.parameters():
                parm.data.clamp_(-clamp_num, clamp_num)
            loss_d, real_score, fake_score = train_discriminator(real_images, real_labels, opt_d)
            # Train generator
            loss_g, loss_c = train_generator(real_labels, opt_g)

        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        losses_c.append(loss_c.cpu().detach().numpy())
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        # Save generated images
        save_samples(epoch + start_idx, fixed_latent)

        # Log losses & scores (last batch)
        print(
            "Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}, loss_s: {:.4f}".format(
                epoch + 1, epochs, loss_g, loss_d, real_score, fake_score, loss_c))

    return losses_g, losses_d, real_scores, fake_scores, losses_c



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", "--learning_rate", help="Select sample_number", type=float, default=0.0002)
    parser.add_argument("-ep", "--epochs", help="Select sample_number", type=int, default=20)
    parser.add_argument("-l", "--lambda_rate", help="Select sample_number", type=int, default=0.1)
    parser.add_argument("-dd", "--data_dir", help="Select sample_number", default='maxwellfdfd')

    args = parser.parse_args()

    lr = args.learning_rate
    epochs = args.epochs
    LAMBDA = args.lambda_rate

    sample_dir = f'generated-lr{lr}-lambda{LAMBDA}-epochs{epochs}'
    os.makedirs(sample_dir, exist_ok=True)

    # sample_eval_dir = f'{sample_dir}/result'

    vid_fname = f'wgan_w_sim_loss_training_{sample_dir}.gif'

    # Data
    data_dir = os.path.join(os.getcwd(), args.data_dir)

    cem_train = CEMDataset(data_dir, train=True, scale=5, is_regression=True)

    batch_size = 128
    train_dl = DataLoader(cem_train, batch_size, shuffle=True, pin_memory=True)

    # GPU
    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)

    # model init

    simulator = CnnModel()
    simulator.load_state_dict(torch.load('cnn_model_ep50.pth'))
    simulator.eval()

    discriminator = nn.Sequential(
        # in: 1 x 20 x 40

        nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=(0, 1), bias=False),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(64, 128, kernel_size=(3, 4), stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(128, 256, kernel_size=(3, 4), stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(256, 1, kernel_size=(3, 5), stride=1, padding=0, bias=False),
        # out: 1 x 1 x 1

        nn.Flatten(),
        nn.Sigmoid()

    )
    print(discriminator)
    simulator = to_device(simulator, device)
    discriminator = to_device(discriminator, device)

    generator = nn.Sequential(
        # in: latent_size x 1 x 1
        nn.ConvTranspose2d(latent_size + NUM_LABELS, 256, kernel_size=(3, 5), stride=1, padding=0, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(True),

        nn.ConvTranspose2d(256, 128, kernel_size=(3, 4), stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(True),

        nn.ConvTranspose2d(128, 64, kernel_size=(3, 4), stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(True),

        nn.ConvTranspose2d(64, 1, kernel_size=(4, 4), stride=2, padding=(0, 1), bias=False),
        nn.Tanh()
        # out: 1 x 20 x 40
    )
    discriminator.apply(weight_init)
    generator.apply(weight_init)

    generator = to_device(generator, device)

    # Test Label
    fixed_label = F.one_hot(
        torch.from_numpy(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] * NUM_SAMPLES_PER_LABEL))).to(device)
    z = torch.randn(NUM_LABELS * NUM_SAMPLES_PER_LABEL, latent_size).to(device)
    fixed_latent = torch.cat((z, fixed_label), 1)
    fixed_latent = fixed_latent[:, :, None, None]
    # save_samples(0, fixed_latent, show=False)

    history = fit(epochs, lr)
    losses_g, losses_d, real_scores, fake_scores, losses_c = history
    # Save the model checkpoints
    torch.save(generator.state_dict(), 'G.pth')
    torch.save(discriminator.state_dict(), 'D.pth')


    files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if 'generated' in f]
    files.sort()

    images = []
    for filename in files:
        images.append(imageio.imread(filename))
    imageio.mimsave(vid_fname, images)

    # save loss curve
    plt.clf()
    plt.plot(losses_d, '-')
    plt.plot(losses_g, '-')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Discriminator', 'Generator'])
    plt.title('Losses')
    plt.savefig(f'loss-{sample_dir}.png', dpi=300)
    plt.clf()

    # save scores
    #print('real_scores', real_scores[0])
    #print('real_scores', fake_scores[0])
    plt.plot(real_scores, '-')
    plt.plot(fake_scores, '-')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.legend(['Real', 'Fake'])
    plt.title('Scores')
    plt.savefig(f'scores-{sample_dir}.png', dpi=300)