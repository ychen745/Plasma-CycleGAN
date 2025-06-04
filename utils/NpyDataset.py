import os
import numpy as np
import random
import scipy.ndimage.interpolation as interpolation
import scipy.ndimage as ndimage
import scipy
import torch
import torch.utils.data
import pandas as pd


def rotate_image(image, angle, axes):
    return ndimage.rotate(image, angle, axes, reshape=False, order=3, mode='constant', cval=0.0)


def rotation3d_image(image, theta_x, theta_y, theta_z):
    rotated_image = rotate_image(image, theta_x, (1, 2))
    rotated_image = rotate_image(image, theta_y, (0, 2))
    rotated_image = rotate_image(image, theta_z, (0, 1))

    return rotated_image

def flip_image(image, axis):
    if axis == 0:
        return image[::-1, :, :]
    elif axis == 1:
        return image[:, ::-1, :]
    elif axis == 2:
        return image[:, :, ::-1]

    print('Flipping: invalid axis.')
    return image


def brightness(image):
    max_val = np.max(image)
    min_val = np.min(image)

    c = np.random.randint(-20, 20)

    new_image = image + c
    new_image = np.clip(new_image, 0, 255)

    return new_image


def contrast(image):
    luminanza = int(np.sum(image) / image.size)

    c = np.random.randint(-20, 20)
    d = image - luminanza
    dc = d * c / 100

    new_image = image + dc
    new_image = np.clip(new_image, 0, 255)

    return new_image


def imadjust(image, gamma=np.random.uniform(1, 2)):
    max_val = np.max(image)
    min_val = np.min(image)

    new_image = (((image - min_val) / (max_val - min_val)) ** gamma) * 255
    new_image = np.clip(new_image, 0, 255)

    return new_image

# --------------------------------------------------------------------------------------


class NpyDataSet(torch.utils.data.Dataset):

    def __init__(self, data_path, data_split='data_split.csv', test_group=4, transforms=None, mode='train'):

        # Init membership variables
        self.data_path = data_path

        df = pd.read_csv(os.path.join(data_path, data_split))
        train_images = df[df['group'] != test_group]['mri_id']
        train_labels = df[df['group'] != test_group]['pet_id']
        test_images = df[df['group'] == test_group]['mri_id']
        test_labels = df[df['group'] == test_group]['pet_id']

        self.train_images_list = [os.path.join(data_path, 'images_npy', ele.split('.')[0] + '.npy') for ele in train_images]
        self.train_labels_list = [os.path.join(data_path, 'labels_npy', ele.split('.')[0] + '.npy') for ele in train_labels]
        self.test_images_list = [os.path.join(data_path, 'images_npy', ele.split('.')[0] + '.npy') for ele in test_images]
        self.test_labels_list = [os.path.join(data_path, 'labels_npy', ele.split('.')[0] + '.npy') for ele in test_labels]

        self.train_images_size = len(self.train_images_list)
        self.train_labels_size = len(self.train_labels_list)
        self.test_images_size = len(self.test_images_list)
        self.test_labels_size = len(self.test_labels_list)

        self.transforms = transforms
        self.mode = mode

    def read_image(self, path):
        image = np.load(path)
        return image

    def __getitem__(self, index):

        if self.mode == 'train':
            data_path = self.train_images_list[index]
            label_path = self.train_labels_list[index]
        else:
            data_path = self.test_images_list[index]
            label_path = self.test_labels_list[index]

        # read image and label
        image = np.load(data_path).astype(np.float32)
        image = Normalization(image)  # set intensity 0-255
        label = np.load(label_path).astype(np.float32)
        label = Normalization(label)

        sample = {'image': image, 'label': label}

        if self.transforms:  # apply the transforms to image and label (normalization, resampling, patches)
            for transform in self.transforms:
                sample = transform(sample)

        image_np = np.absolute((sample['image'] - 127.5) / 127.5)
        label_np = np.absolute((sample['label'] - 127.5) / 127.5)

        image_np = image_np[np.newaxis, :, :, :]
        label_np = label_np[np.newaxis, :, :, :]

        return torch.from_numpy(image_np), torch.from_numpy(label_np)  # this is the final output to feed the network

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_images_list)
        else:
            return len(self.test_images_list)


class NpyDataSet_testing(torch.utils.data.Dataset):
    def __init__(self, data_list,label_list,
                 transforms=None,
                 mode='test'):

        # Init membership variables
        self.data_list = data_list
        self.label_list = label_list
        self.transforms = transforms
        self.mode = mode

    def read_image(self, path):
        image = np.load(path)
        return image

    def __getitem__(self, item):

        data_dict = self.data_list[item]
        label_dict = self.label_list[item]
        data_path = data_dict["data"]
        label_path = label_dict["label"]

        data_path = data_path
        label_path = label_path

        # read image and label
        image = self.read_image(data_path)
        image = Normalization(image)  # set intensity 0-255

        label = self.read_image(label_path)
        label = Normalization(label)  # set intensity 0-255

        sample = {'image': image, 'label': label}

        if self.transforms:  # apply the transforms to image and label (normalization, resampling, patches)
            for transform in self.transforms:
                sample = transform(sample)

        image_np = np.absolute((sample['image'] - 127.5) / 127.5)
        label_np = np.absolute((sample['label'] - 127.5) / 127.5)

        image_np = image_np[np.newaxis, :, :, :]
        label_np = label_np[np.newaxis, :, :, :]

        return torch.from_numpy(image_np), torch.from_numpy(label_np)  # this is the final output to feed the network

    def __len__(self):
        return len(self.data_list)


def Normalization(image):
    image *= 255.0 / np.max(image)
    return image


class RandomCrop(object):
    """
    Crop randomly the image in a sample. This is usually used for data augmentation.
      Drop ratio is implemented for randomly dropout crops with empty label. (Default to be 0.2)
      This transformation only applicable in train mode

    Args:
      output_size (tuple or int): Desired output size. If int, cubic crop is made.
    """

    def __init__(self, output_size):
        self.name = 'Random Crop'

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        size_old = image.shape
        size_new = self.output_size

        contain_label = False

        # get the start crop coordinate in ijk
        if size_old[0] <= size_new[0]:
            start_i = 0
        else:
            start_i = np.random.randint(0, size_old[0] - size_new[0])

        if size_old[1] <= size_new[1]:
            start_j = 0
        else:
            start_j = np.random.randint(0, size_old[1] - size_new[1])

        if size_old[2] <= size_new[2]:
            start_k = 0
        else:
            start_k = np.random.randint(0, size_old[2] - size_new[2])

        new_image = image[start_i:start_i + size_new[0], start_j:start_j + size_new[1], start_k:start_k + size_new[2]]
        new_label = label[start_i:start_i + size_new[0], start_j:start_j + size_new[1], start_k:start_k + size_new[2]]

        return {'image': new_image, 'label': new_label}


class Augmentation(object):
    """
    Application of transforms. This is usually used for data augmentation.
    List of transforms: random noise
    """

    def __init__(self):
        self.name = 'Augmentation'

    def __call__(self, sample):

        choice = np.random.choice([0, 1, 2, 3, 4, 5, 6])

        # no augmentation
        if choice == 0:  # no augmentation

            image, label = sample['image'], sample['label']
            return {'image': image, 'label': label}

        # Additive Gaussian noise
        if choice == 1:  # Additive Gaussian noise
            image, label = sample['image'], sample['label']
            mean = np.random.uniform(0, 1)
            std = np.random.uniform(0, 2)
            noise = np.random.normal(mean, std, image.shape).astype(np.float32)

            new_image = image + noise
            new_label = label + noise
            new_image = np.clip(new_image, 0, 255)
            new_label = np.clip(new_label, 0, 255)

            return {'image': new_image, 'label': new_label}


        # Random rotation x y z
        if choice == 2:  # Random rotation

            theta_x = np.random.randint(-180, 180)
            theta_y = np.random.randint(-180, 180)
            theta_z = np.random.randint(-180, 180)
            image, label = sample['image'], sample['label']

            image = rotation3d_image(image,theta_x,theta_y, theta_z)
            label = rotation3d_image(label,theta_x,theta_y, theta_z)

            return {'image': image, 'label': label}


        # Random flip
        if choice == 3:  # Random flip

            axis = np.random.choice([0, 1, 2])
            image, label = sample['image'], sample['label']

            image = flip_image(image, axis)
            label = flip_image(label, axis)

            return {'image': image, 'label': label}

        # Brightness
        if choice == 4:  # Brightness

            image, label = sample['image'], sample['label']

            image = brightness(image)
            label = brightness(label)

            return {'image': image, 'label': label}

        # Contrast
        if choice == 5:  # Contrast

            image, label = sample['image'], sample['label']

            image = contrast(image)
            label = contrast(label)

            return {'image': image, 'label': label}


        # histogram gamma
        if choice == 6:
            image, label = sample['image'], sample['label']

            image = imadjust(image)
            label = imadjust(label)

            return {'image': image, 'label': label}
