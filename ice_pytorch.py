import os
import numbers
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import Compose, RandomVerticalFlip, RandomHorizontalFlip, Normalize, ToTensor, RandomRotation, ToPILImage
from torch.autograd import Variable
from net_pytorch import MyNet
use_cuda = torch.cuda.is_available()

print("Use GPU" if use_cuda else "CPU")

# Constants
train_file = os.path.join(os.getcwd(), "./input/train.json")
test_file = os.path.join(os.getcwd(), "./input/test.json")
model_file = os.path.join(os.getcwd(), "./model_checkpoint.hdf5")
model_json_file = os.path.join(os.getcwd(), "./model.json")
logs_dir = os.path.join(os.getcwd(), "./logs/")


# Preparing data
data = pd.read_json(train_file)
data['band_1'] = data['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
data['band_2'] = data['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
data['inc_angle'] = pd.to_numeric(data['inc_angle'], errors='coerce')


band_1 = np.concatenate([im for im in data['band_1']]).reshape(-1, 75, 75)
band_2 = np.concatenate([im for im in data['band_2']]).reshape(-1, 75, 75)
# band_3 = band_1 + band_2
band_1 = (band_1 - band_1.mean()) / band_1.std()
band_2 = (band_2 - band_2.mean()) / band_2.std()
angles = data['inc_angle'].values

image = np.stack([band_1, band_2], axis=-1)


##########################################################################################################

class IcebergDataset(Dataset):
    """
    Dataset for iceberg file
    Args
    ----------
    bands : (numpy array, any dtype) - image with shape H x W x NUM_CHANNELS
    angles : (numpy array, any dtype) - array of values of angles
    target : (numpy array, any dtype) - array of ones and zeroes
    """

    def __init__(self, bands, angles, target, transform=None):
        # self.img = torch.from_numpy(bands.astype('float32')).type(torch.FloatTensor)
        self.angles = torch.from_numpy(angles.astype(
            'float32')).type(torch.FloatTensor).cuda()
        self.target = torch.from_numpy(target.astype(
            'float32').reshape((target.shape[0], 1))).type(torch.FloatTensor).cuda()
        self.img = bands.astype('float32')
        # self.angles = angles.astype('float32')
        # self.target = target.astype('float32')
        self.length = len(self.img)
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.transform:
            return (self.transform(self.img[idx]), self.angles[idx], self.target[idx])
        else:
            return (self.img[idx], self.angles[idx], self.target[idx])


class CentralCropNumpy(object):
    """Crops the given image in np.array in format WxHxC at center

    Args
    ----------
    size : (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        x, y = img.shape[0], img.shape[1]
        cropx, cropy = self.size
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty:starty + cropy, startx:startx + cropx, :]


def show_two_channels(img, title=None, show=True):
    """
    Shows two-channel images using 2 subplots

    Args
    ----------
    img : image in numpy array
    title : optional, sets title to figure  
    """
    fig = plt.figure()
    if title:
        fig.suptitle(title)
    b1 = fig.add_subplot(121)
    plt.imshow(img[:, :, 0])
    b1.set_title('Band 1')
    b2 = fig.add_subplot(122)
    plt.imshow(img[:, :, 1])
    b2.set_title('Band 2')
    if show:
        plt.show()


#########################################################################################################

means = [(band_1/255).mean(), (band_2/255).mean()]
stds = [(band_1/255).std(), (band_2/255).std()]

transform = Compose([
    CentralCropNumpy(50),
    ToTensor(),
    # Normalize(mean=means,  # Help here pls!!! This doesn't work properly
            #   std=stds),
    # ToPILImage(),
    # RandomHorizontalFlip(),
    # RandomVerticalFlip(),
    # RandomRotation(30),
    # ToTensor()
])

assert image[0].astype('float32')/255 ==  transform(image[0]).numpy() # FAIL

quit()

# Preparing dataloaders
ds = IcebergDataset(image, angles, data['is_iceberg'].values, transform)

split_rate = 0.15
batch_size = 32
num_epoches = 20

train_indices = list(range(0, int(len(ds) * (1 - split_rate))))
test_indices = list(range(train_indices[-1] + 1, len(ds)))

train_dl = DataLoader(dataset=ds, batch_size=batch_size,
                      sampler=SubsetRandomSampler(train_indices), num_workers=0)
test_dl = DataLoader(dataset=ds, batch_size=batch_size,
                     sampler=SubsetRandomSampler(test_indices), num_workers=0)


model = MyNet()

loss_function = torch.nn.BCELoss()

lr = 0.001

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
if use_cuda:
    model.cuda()
    loss_function.cuda()

criterion = loss_function

all_losses = []
val_losses = []

for epoch in range(num_epoches):
    print('Epoch {}'.format(epoch + 1))
    print('*' * 5 + ':')
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_dl, 1):
        img, angle, label = data
        if use_cuda:
            img, angle, label = Variable(img.cuda()), Variable(angle.type(
                torch.FloatTensor).cuda()), Variable(label.type(torch.FloatTensor).cuda())  # On GPU
            # img, angle, label = img.cuda(), angle.cuda(), label.cuda()
        else:
            img, angle, label = Variable(img), Variable(angle), Variable(
                label)

        out = model(img, angle)
        loss = criterion(out, label)
        running_loss += loss.data[0] * label.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            all_losses.append(running_loss / (batch_size * i))
            print('[{}/{}] Loss: {:.6f}'.format(
                epoch + 1, num_epoches, running_loss / (batch_size * i),
                running_acc / (batch_size * i)))

    print('Finish {} epoch, Loss: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_dl))))

    model.eval()
    eval_loss = 0
    eval_acc = 0
    for data in test_dl:
        img, angle, label = data

        if use_cuda:
            img, angle, label = Variable(img.cuda(), volatile=True), Variable(angle.type(torch.FloatTensor).cuda(
            ), volatile=True), Variable(label.type(torch.FloatTensor).cuda(), volatile=True)  # On GPU
        else:
            img = Variable(img, volatile=True)
            angle = Variable(angle, volatile=True)
            label = Variable(label, volatile=True)

        out = model(img, angle)
        loss = criterion(out, label)
        eval_loss += loss.data[0] * label.size(0)

    print('VALIDATION Loss: {:.6f}'.format(eval_loss / (len(test_dl))))
    val_losses.append(eval_loss / (len(test_dl)))
    print()
