# -*- coding: utf-8 -*
"""This module is served as torchvision.datasets to load CUB200-2011.

CUB200-2011 dataset has 11,788 images of 200 bird species. The project page
is as follows.
    http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
- Images are contained in the directory data/cub200/raw/images/,
  with 200 subdirectories.
- Format of images.txt: <image_id> <image_name>
- Format of train_test_split.txt: <image_id> <is_training_image>
- Format of classes.txt: <class_id> <class_name>
- Format of iamge_class_labels.txt: <image_id> <class_id>

This file is modified from:
    https://github.com/vishwakftw/vision.
"""


import os
import pickle

import numpy as np
import PIL.Image
import torch


# coarse_class = "dogs"
# from testing_fine_grained import coarse_class

__all__ = ['CUB200']
__author__ = 'Hao Zhang'
__copyright__ = '2018 LAMDA'
__date__ = '2018-01-09'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2018-01-10'
__version__ = '1.0'


class CUB200(torch.utils.data.Dataset):
    """CUB200 dataset.

    Args:
        _root, str: Root directory of the dataset.
        _train, bool: Load train/test data.
        _transform, callable: A function/transform that takes in a PIL.Image
            and transforms it.
        _target_transform, callable: A function/transform that takes in the
            target and transforms it.
        _train_data, list of np.ndarray.
        _train_labels, list of int.
        _test_data, list of np.ndarray.
        _test_labels, list of int.
    """
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, coarse_class=None):
        """Load the dataset.

        Args
            root, str: Root directory of the dataset.
            train, bool [True]: Load train/test data.
            transform, callable [None]: A function/transform that takes in a
                PIL.Image and transforms it.
            target_transform, callable [None]: A function/transform that takes
                in the target and transforms it.
            download, bool [False]: If true, downloads the dataset from the
                internet and puts it in root directory. If dataset is already
                downloaded, it is not downloaded again.
        """
#         self._root = os.path.expanduser(root)  # Replace ~ by the complete dir
#         self._train = train
        self._transform = transform
        self._target_transform = target_transform
        self._coarse_class = coarse_class

#         if self._checkIntegrity():
#             print('Files already downloaded and verified.')
# #         elif download:
# #             url = ('http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/'
# #                    'CUB_200_2011.tgz')
# #             self._download(url)
# #             self._extract()
#         else:
        self._extract()
#             raise RuntimeError(
#                 'Dataset not found. You can use download=True to download it.')

        # Now load the picked data.
#         if self._train:
#             self._train_data, self._train_labels = pickle.load(open(
#                 os.path.join(self._root, 'processed/train.pkl'), 'rb'))
# #             assert (len(self._train_data) == 5994
# #                     and len(self._train_labels) == 5994)
#             assert (len(self._train_data) == 513
#                     and len(self._train_labels) == 513) #change
#         else:
#             self._test_data, self._test_labels = pickle.load(open(
#                 os.path.join(self._root, 'processed/test.pkl'), 'rb'))
# #             assert (len(self._test_data) == 5794
# #                     and len(self._test_labels) == 5794)
#             assert (len(self._test_data) == 57
#                     and len(self._test_labels) == 57) #change
#         print("dsddsd")
        base_path = "data/testing/" + self._coarse_class
        self._test_data = pickle.load(open(base_path + '/test.pkl', 'rb'))
#             assert (len(self._test_data) == 5794
#                     and len(self._test_labels) == 5794)
#         assert (len(self._test_data) == 57
#                 and len(self._test_labels) == 57) #change


    def __getitem__(self, index):
        """
        Args:
            index, int: Index.

        Returns:
            image, PIL.Image: Image of the given index.
            target, str: target of the given index.
        """
#         if self._train:
#             image, target = self._train_data[index], self._train_labels[index]
#         else:
#             image, target = self._test_data[index], self._test_labels[index]
        image = self._test_data[index]
        # Doing this so that it is consistent with all other datasets.
        image = PIL.Image.fromarray(image)

        if self._transform is not None:
            image = self._transform(image)
#         if self._target_transform is not None:
#             target = self._target_transform(target)

        return image

    def __len__(self):
        """Length of the dataset.

        Returns:
            length, int: Length of the dataset.
        """
#         if self._train:
#             return len(self._train_data)
        return len(self._test_data)

    def _checkIntegrity(self):
        """Check whether we have already processed the data.

        Returns:
            flag, bool: True if we have already processed the data.
        """
        base_path = "data/testing/" + self._coarse_class
        return os.path.isfile(base_path + "/test.pkl")

    def _download(self, url):
        """Download and uncompress the tar.gz file from a given URL.

        Args:
            url, str: URL to be downloaded.
        """
        return #change
    
        import six.moves
        import tarfile

        raw_path = os.path.join(self._root, 'raw')
        processed_path = os.path.join(self._root, 'processed')
        if not os.path.isdir(raw_path):
            os.mkdir(raw_path, mode=0o775)
        if not os.path.isdir(processed_path):
            os.mkdir(processed_path, mode=0o775)

        # Downloads file.
        fpath = os.path.join(self._root, 'raw/CUB_200_2011.tgz')
        try:
            print('Downloading ' + url + ' to ' + fpath)
#             six.moves.urllib.request.urlretrieve(url, fpath)
            print("Downloaded!!")
        except six.moves.urllib.error.URLError:
            if url[:5] == 'https:':
                self._url = self._url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.')
                print('Downloading ' + url + ' to ' + fpath)
                six.moves.urllib.request.urlretrieve(url, fpath)

        # Extract file.
#         cwd = os.getcwd()
#         tar = tarfile.open(fpath, 'r:gz')
#         os.chdir(os.path.join(self._root, 'raw'))
#         tar.extractall()
#         tar.close()
#         os.chdir(cwd)_extract

    def _extract(self):
        test_data = []
#         print("dasds")
        base_path = "data/testing/" + self._coarse_class
        my_images = os.listdir(base_path)
        my_images = [i for i in my_images if ".jpg" in i]
        my_images.sort()
#         print(len(my_images))
        pickle.dump(my_images, open(base_path + '/test_file_names.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        for i in my_images:
            if ".jpg" not in i:
                continue
            image = PIL.Image.open(os.path.join(base_path, i))
            if image.getbands()[0] == 'L':
                image = image.convert('RGB')
            image_np = np.array(image)
            image.close()
            test_data.append(image_np)
#         print("data" , len(test_data))
        """Prepare the data for train/test split and save onto disk."""
#         image_path = os.path.join(self._root, 'raw/CUB_200_2011/images/')
#         image_path = "data/dogs/raw/dogs_/images"
        # Format of images.txt: <image_id> <image_name>
#         id2name = np.genfromtxt(os.path.join(
#             self._root, 'raw/CUB_200_2011/images.txt'), dtype=str)
#         id2name = np.genfromtxt("data/dogs/raw/dogs_/images.txt", dtype=str)
        # Format of train_test_split.txt: <image_id> <is_training_image>
#         id2train = np.genfromtxt(os.path.join(
#             self._root, 'raw/CUB_200_2011/train_test_split.txt'), dtype=int)

#         id2train = np.genfromtxt("data/dogs/raw/dogs_/train_test_split.txt", dtype=int)

#         train_data = []
#         train_labels = []
        
#         test_labels = []
#         for id_ in range(id2name.shape[0]):
#             image = PIL.Image.open(os.path.join(image_path, id2name[id_, 1]))
#             label = int(id2name[id_, 1][:3]) - 1  # Label starts with 0

#             # Convert gray scale image to RGB image.
#             if image.getbands()[0] == 'L':
#                 image = image.convert('RGB')
#             image_np = np.array(image)
#             image.close()

#             if id2train[id_, 1] == 1:
#                 train_data.append(image_np)
#                 train_labels.append(label)
#             else:
#                 test_data.append(image_np)
#                 test_labels.append(label)
         
         
#         pickle.dump((train_data, train_labels),
#                     open('data/cub200/processed/train.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
                    
#         pickle.dump((test_data, test_labels),
#                     open('data/cub200/processed/test.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        
#         pickle.dump((train_data, train_labels),
#                     open('data/dogs/processed/train.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
                    
        pickle.dump(test_data,
                    open(base_path + '/test.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
