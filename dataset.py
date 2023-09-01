
"""Data Processing for StarGAN"""
import os
import math
import random
import multiprocessing
import numpy as np
from PIL import Image

import mindspore.dataset.vision.py_transforms as py_vision
import mindspore.dataset.transforms.py_transforms as py_transforms
import mindspore.dataset as de


def is_image_file(filename):
    """Judge whether it is an image"""
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif', '.tiff']
    return any(filename.lower().endswith(extension) for extension in IMG_EXTENSIONS)



class CelebA:
    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(1.0 if values[idx] == '1' else 0.0)

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, idx):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[idx]
        image = np.asarray(Image.open(os.path.join(self.image_dir, filename)))
        label = np.asarray(label)
        image = np.squeeze(self.transform(image))
        return image, label

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(data_root, attr_path, selected_attrs, crop_size=178, image_size=128,
               dataset='CelebA', mode='train'):
    """Build and return a data loader."""
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    transform = [py_vision.ToPIL()]
    if mode == 'train':
        transform.append(py_vision.RandomHorizontalFlip())
        transform.append(py_vision.CenterCrop(crop_size))
    transform.append(py_vision.Resize([image_size, image_size]))
    transform.append(py_vision.ToTensor())
    transform.append(py_vision.Normalize(mean=mean, std=std))
    transform = py_transforms.Compose(transform)

    dataset = CelebA(data_root, attr_path, selected_attrs, transform, mode)

    return dataset


def dataloader(img_path, attr_path, selected_attr, dataset, mode='train',
               batch_size=1, device_num=4, rank=0, shuffle=True):
    """Get dataloader"""
    #assert dataset in ['CelebA']

    cores = multiprocessing.cpu_count()
    num_parallel_workers = int(cores / device_num)

    
    dataset_loader = get_loader(img_path, attr_path, selected_attr, mode=mode)
    length_dataset = len(dataset_loader)
    distributed_sampler = DistributedSampler(length_dataset, device_num, rank, shuffle=shuffle)
    dataset_column_names = ["image", "attr"]

    if device_num != 8:
        ds = de.GeneratorDataset(dataset_loader, column_names=dataset_column_names,
                                 num_parallel_workers=min(32, num_parallel_workers),
                                 sampler=distributed_sampler)
        ds = ds.batch(batch_size, num_parallel_workers=min(32, num_parallel_workers), drop_remainder=True)

    else:
        ds = de.GeneratorDataset(dataset_loader, column_names=dataset_column_names, sampler=distributed_sampler)
        ds = ds.batch(batch_size, num_parallel_workers=min(8, num_parallel_workers), drop_remainder=True)
    if mode == 'train':
        ds = ds.repeat(200)

    return ds, length_dataset


class DistributedSampler:
    """Distributed sampler."""
    def __init__(self, dataset_size, num_replicas=None, rank=None, shuffle=False):
        if num_replicas is None:
            print("***********Setting world_size to 1 since it is not passed in ******************")
            num_replicas = 1
        if rank is None:
            print("***********Setting rank to 0 since it is not passed in ******************")
            rank = 0

        self.dataset_size = dataset_size
        self.num_replicas = num_replicas
        self.epoch = 0
        self.rank = rank
        self.num_samples = int(math.ceil(dataset_size * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # shuffle based on epoch
        if self.shuffle:
            indices = np.random.RandomState(seed=self.epoch).permutation(self.dataset_size)
            indices = indices.tolist()
            self.epoch += 1
        else:
            indices = list(range(self.dataset_size))

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples
