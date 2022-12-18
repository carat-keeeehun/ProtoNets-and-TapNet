import math
import os
import pickle
import shutil
import warnings
from glob import glob

import torch
from torch.utils.data import Dataset
from torchvision import datasets as dset
from torchvision import transforms
from temp import _get_confirm_token, _save_response_content

from PIL import Image
import numpy as np

import cv2
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")

# BASEDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
BASEDIR = "/home/keeeehun/source/meta"

def get_data_dir():
    # return os.path.join(BASEDIR, 'data')
    path = "/home/keeeehun/source/meta/data"
    return path

def pairwise_distances(x, y, matching_fn):
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.
    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn.lower() == 'l2' or matching_fn.lower == 'euclidean':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances
    elif matching_fn.lower() == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn.lower() == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise (ValueError('Unsupported similarity function'))

def margin_of_error(values, confidence_interval=1.96):
    num = len(values)
    mean = sum(values) / num
    variance = sum(list(map(lambda x: pow(x - mean, 2), values))) / num

    standard_deviation = math.sqrt(variance)
    standard_error = standard_deviation / math.sqrt(num)

    return mean, standard_error * confidence_interval

class MiniImageNetDataset(Dataset):
    def __init__(self, mode='train', transform=None):
        super().__init__()
        self.root_dir = BASEDIR + '/data/miniImageNet'

        if transform is None:
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = transform

        if not os.path.exists(self.root_dir):
            print('Data not found. Downloading data')
            self.download()
        if mode == 'matching_train':
            import numpy as np
            dataset_train = pickle.load(open(os.path.join(self.root_dir, 'train'), 'rb'))
            dataset_val = pickle.load(open(os.path.join(self.root_dir, 'val'), 'rb'))

            image_data_train = dataset_train['image_data']
            class_dict_train = dataset_train['class_dict']
            image_data_val = dataset_val['image_data']
            class_dict_val = dataset_val['class_dict']

            image_data = np.concatenate((image_data_train, image_data_val), axis=0)
            class_dict = class_dict_train.copy()
            class_dict.update(class_dict_val)
            dataset = {'image_data': image_data, 'class_dict': class_dict}
        else:
            dataset = pickle.load(open(os.path.join(self.root_dir, mode), 'rb'))

        self.x = dataset['image_data']

        self.y = torch.arange(len(self.x))
        for idx, (name, id) in enumerate(dataset['class_dict'].items()):
            if idx > 63:
                id[0] = id[0] + 38400
                id[-1] = id[-1] + 38400
            s = slice(id[0], id[-1] + 1)
            self.y[s] = idx

    def __getitem__(self, index):
        img = self.x[index]
        x = self.transform(image=img)['image']

        return x, self.y[index]

    def __len__(self):
        return len(self.x)

    def download(self):
        import tarfile
        gdrive_id = '16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY'
        gz_filename = 'mini-imagenet.tar.gz'
        root = BASEDIR + '/data/miniImageNet'

        self.download_file_from_google_drive(gdrive_id, root, gz_filename)

        filename = os.path.join(root, gz_filename)

        with tarfile.open(filename, 'r') as f:
            f.extractall(root)

        os.rename(BASEDIR + '/data/miniImageNet/mini-imagenet-cache-train.pkl', BASEDIR + '/data/miniImageNet/train')
        os.rename(BASEDIR + '/data/miniImageNet/mini-imagenet-cache-val.pkl', BASEDIR + '/data/miniImageNet/val')
        os.rename(BASEDIR + '/data/miniImageNet/mini-imagenet-cache-test.pkl', BASEDIR + '/data/miniImageNet/test')

    def download_file_from_google_drive(self, file_id, root, filename):

        """Download a Google Drive file from  and place it in root.
        Args:
            file_id (str): id of file to be downloaded
            root (str): Directory to place downloaded file in
            filename (str, optional): Name to save the file under. If None, use the id of the file.
            md5 (str, optional): MD5 checksum of the download. If None, do not check
        """
        # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
        import requests
        url = "https://docs.google.com/uc?export=download"

        root = os.path.expanduser(root)
        if not filename:
            filename = file_id
        fpath = os.path.join(root, filename)

        os.makedirs(root, exist_ok=True)

        if os.path.isfile(fpath):
            print('Using downloaded and verified file: ' + fpath)
        else:
            session = requests.Session()

            response = session.get(url, params={'id': file_id}, stream=True)
            token = _get_confirm_token(response)

            if token:
                params = {'id': file_id, 'confirm': token}
                response = session.get(url, params=params, stream=True)

            _save_response_content(response, fpath)


class TieredImageNet(Dataset):

    def __init__(self, split='train', mini=False, **kwargs):
        root_path = BASEDIR + '/data/tieredImageNet'
        split_tag = split
        data = np.load(os.path.join(
                root_path, '{}_images.npz'.format(split_tag)),
                allow_pickle=True)['images']
        data = data[:, :, :, ::-1]
        with open(os.path.join(
                root_path, '{}_labels.pkl'.format(split_tag)), 'rb') as f:
            label = pickle.load(f)['labels']

        image_size = 80
        data = [Image.fromarray(x) for x in data]

        min_label = min(label)
        label = [x - min_label for x in label]

        if mini:
            data_ = []
            label_ = []
            np.random.seed(0)
            c = np.random.choice(max(label) + 1, 64, replace=False).tolist()
            n = len(data)
            cnt = {x: 0 for x in c}
            ind = {x: i for i, x in enumerate(c)}
            for i in range(n):
                y = int(label[i])
                if y in c and cnt[y] < 600:
                    data_.append(data[i])
                    label_.append(ind[y])
                    cnt[y] += 1
            data = data_
            label = label_

        self.x = data
        self.y = torch.tensor(label)
        self.n_classes = max(self.y) + 1

        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
        normalize = transforms.Normalize(**norm_params)
        self.default_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize,  
        ])
        augment = kwargs.get('augment')
        if augment == 'resize':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment == 'crop':
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment is None:
            self.transform = self.default_transform

        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return x * std + mean
        self.convert_raw = convert_raw

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.transform(self.x[i]), self.y[i]