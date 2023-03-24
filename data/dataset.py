import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import pickle
import numpy as np
from PIL import Image
from pathlib import Path


def get_transform(grayscale=False, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)


class TextDataset:

    def __init__(self, base_path, collator_resolution, num_examples=15, target_transform=None, min_virtual_size=0):
        self.NUM_EXAMPLES = num_examples
        self.min_virtual_size = min_virtual_size

        # base_path=DATASET_PATHS
        file_to_store = open(base_path, "rb")
        self.IMG_DATA = pickle.load(file_to_store)['train']
        self.IMG_DATA = dict(list(self.IMG_DATA.items()))  # [:NUM_WRITERS])
        if 'None' in self.IMG_DATA.keys():
            del self.IMG_DATA['None']

        self.alphabet = ''.join(sorted(set(''.join(d['label'] for d in sum(self.IMG_DATA.values(), [])))))
        self.author_id = list(self.IMG_DATA.keys())

        self.transform = get_transform(grayscale=True)
        self.target_transform = target_transform

        self.collate_fn = TextCollator(collator_resolution)

    def __len__(self):
        return max(len(self.author_id), self.min_virtual_size)

    @property
    def num_writers(self):
        return len(self.author_id)

    def __getitem__(self, index):
        NUM_SAMPLES = self.NUM_EXAMPLES
        index = index % len(self.author_id)

        author_id = self.author_id[index]

        self.IMG_DATA_AUTHOR = self.IMG_DATA[author_id]
        random_idxs = np.random.choice(len(self.IMG_DATA_AUTHOR), NUM_SAMPLES, replace=True)

        rand_id_real = np.random.choice(len(self.IMG_DATA_AUTHOR))
        real_img = self.transform(self.IMG_DATA_AUTHOR[rand_id_real]['img'].convert('L'))
        real_labels = self.IMG_DATA_AUTHOR[rand_id_real]['label'].encode()

        imgs = [np.array(self.IMG_DATA_AUTHOR[idx]['img'].convert('L')) for idx in random_idxs]
        labels = [self.IMG_DATA_AUTHOR[idx]['label'].encode() for idx in random_idxs]

        max_width = 192  # [img.shape[1] for img in imgs]

        imgs_pad = []
        imgs_wids = []

        for img in imgs:
            img = 255 - img
            img_height, img_width = img.shape[0], img.shape[1]
            outImg = np.zeros((img_height, max_width), dtype='float32')
            outImg[:, :img_width] = img[:, :max_width]

            img = 255 - outImg

            imgs_pad.append(self.transform(Image.fromarray(img.astype(np.uint8))))
            imgs_wids.append(img_width)

        imgs_pad = torch.cat(imgs_pad, 0)

        item = {
            'simg': imgs_pad,  # widths of the N images [list(N)]
            'swids': imgs_wids,  # N images (15) that come from the same author [N (15), H (32), MAX_W (192)]
            'img': real_img,  # the input image [1, H (32), W]
            'label': real_labels,  # the label of the input image [byte]
            'img_path': 'img_path',
            'idx': 'indexes',
            'wcl': index  # id of the author [int]
        }
        return item


class TextDatasetval():

    def __init__(self, base_path, collator_resolution, num_examples=15, target_transform=None, min_virtual_size=0):
        self.NUM_EXAMPLES = num_examples
        self.min_virtual_size = min_virtual_size

        # base_path=DATASET_PATHS
        file_to_store = open(base_path, "rb")
        self.IMG_DATA = pickle.load(file_to_store)['test']
        self.IMG_DATA = dict(list(self.IMG_DATA.items()))  # [NUM_WRITERS:])
        if 'None' in self.IMG_DATA.keys():
            del self.IMG_DATA['None']
        self.alphabet = ''.join(sorted(set(''.join(d['label'] for d in sum(self.IMG_DATA.values(), [])))))
        self.author_id = list(self.IMG_DATA.keys())

        self.transform = get_transform(grayscale=True)
        self.target_transform = target_transform

        self.collate_fn = TextCollator(collator_resolution)

    def __len__(self):
        return max(len(self.author_id), self.min_virtual_size)

    @property
    def num_writers(self):
        return len(self.author_id)

    def __getitem__(self, index):
        NUM_SAMPLES = self.NUM_EXAMPLES
        index = index % len(self.author_id)

        author_id = self.author_id[index]

        self.IMG_DATA_AUTHOR = self.IMG_DATA[author_id]
        random_idxs = np.random.choice(len(self.IMG_DATA_AUTHOR), NUM_SAMPLES, replace=True)

        rand_id_real = np.random.choice(len(self.IMG_DATA_AUTHOR))
        real_img = self.transform(self.IMG_DATA_AUTHOR[rand_id_real]['img'].convert('L'))
        real_labels = self.IMG_DATA_AUTHOR[rand_id_real]['label'].encode()

        imgs = [np.array(self.IMG_DATA_AUTHOR[idx]['img'].convert('L')) for idx in random_idxs]
        labels = [self.IMG_DATA_AUTHOR[idx]['label'].encode() for idx in random_idxs]

        max_width = 192  # [img.shape[1] for img in imgs]

        imgs_pad = []
        imgs_wids = []

        for img in imgs:
            img = 255 - img
            img_height, img_width = img.shape[0], img.shape[1]
            outImg = np.zeros((img_height, max_width), dtype='float32')
            outImg[:, :img_width] = img[:, :max_width]

            img = 255 - outImg

            imgs_pad.append(self.transform(Image.fromarray(img.astype(np.uint8))))
            imgs_wids.append(img_width)

        imgs_pad = torch.cat(imgs_pad, 0)

        item = {'simg': imgs_pad, 'swids': imgs_wids, 'img': real_img, 'label': real_labels, 'img_path': 'img_path',
                'idx': 'indexes', 'wcl': index}
        return item


class TextCollator(object):
    def __init__(self, resolution):
        self.resolution = resolution

    def __call__(self, batch):
        if isinstance(batch[0], list):
            batch = sum(batch, [])
        img_path = [item['img_path'] for item in batch]
        width = [item['img'].shape[2] for item in batch]
        indexes = [item['idx'] for item in batch]
        simgs = torch.stack([item['simg'] for item in batch], 0)
        wcls = torch.Tensor([item['wcl'] for item in batch])
        swids = torch.Tensor([item['swids'] for item in batch])
        imgs = torch.ones([len(batch), batch[0]['img'].shape[0], batch[0]['img'].shape[1], max(width)],
                          dtype=torch.float32)
        for idx, item in enumerate(batch):
            try:
                imgs[idx, :, :, 0:item['img'].shape[2]] = item['img']
            except:
                print(imgs.shape)
        item = {'img': imgs, 'img_path': img_path, 'idx': indexes, 'simg': simgs, 'swids': swids, 'wcl': wcls}
        if 'label' in batch[0].keys():
            labels = [item['label'] for item in batch]
            item['label'] = labels
        if 'z' in batch[0].keys():
            z = torch.stack([item['z'] for item in batch])
            item['z'] = z
        return item


class CollectionTextDataset(Dataset):
    def __init__(self, datasets, datasets_path, dataset_class, **kwargs):
        self.datasets = {}
        for dataset_name in sorted(datasets.split(',')):
            dataset = dataset_class(os.path.join(datasets_path, f'{dataset_name}-32.pickle'), **kwargs)
            self.datasets[dataset_name] = dataset
        self.alphabet = ''.join(sorted(set(''.join(d.alphabet for d in self.datasets.values()))))

    def __len__(self):
        return sum(len(d) for d in self.datasets.values())

    @property
    def num_writers(self):
        return sum(d.num_writers for d in self.datasets.values())

    def __getitem__(self, index):
        for dataset in self.datasets.values():
            if index < len(dataset):
                return dataset[index]
            index -= len(dataset)
        raise IndexError

    def get_dataset(self, index):
        for dataset_name, dataset in self.datasets.items():
            if index < len(dataset):
                return dataset_name
            index -= len(dataset)
        raise IndexError

    def collate_fn(self, batch):
        return self.datasets[self.get_dataset(0)].collate_fn(batch)

class FidDataset:
    def __init__(self, base_path, collator_resolution, num_examples=15, target_transform=None, mode='train'):
        self.NUM_EXAMPLES = num_examples

        # base_path=DATASET_PATHS
        with open(base_path, "rb") as f:
            self.IMG_DATA = pickle.load(f)
        self.IMG_DATA = self.IMG_DATA[mode]
        # self.IMG_DATA = dict(list(self.IMG_DATA.items()))  # [:NUM_WRITERS])
        if 'None' in self.IMG_DATA.keys():
            del self.IMG_DATA['None']

        self.alphabet = ''.join(sorted(set(''.join(d['label'] for d in sum(self.IMG_DATA.values(), [])))))
        self.author_id = sorted(self.IMG_DATA.keys())

        self.transform = get_transform(grayscale=True)
        self.target_transform = target_transform
        self.dataset_size = sum(len(samples) for samples in self.IMG_DATA.values())
        self.collate_fn = TextCollator(collator_resolution)

    def __len__(self):
        return self.dataset_size

    @property
    def num_writers(self):
        return len(self.author_id)

    def __getitem__(self, index):
        NUM_SAMPLES = self.NUM_EXAMPLES
        sample, author_id = None, None
        for author_id, samples in self.IMG_DATA.items():
            if index < len(samples):
                sample, author_id = samples[index], author_id
                break
            index -= len(samples)

        self.IMG_DATA_AUTHOR = self.IMG_DATA[author_id]
        random_idxs = np.random.choice(len(self.IMG_DATA_AUTHOR), NUM_SAMPLES, replace=True)

        real_img = self.transform(sample['img'].convert('L'))
        real_labels = sample['label'].encode()

        imgs = [np.array(self.IMG_DATA_AUTHOR[idx]['img'].convert('L')) for idx in random_idxs]
        labels = [self.IMG_DATA_AUTHOR[idx]['label'].encode() for idx in random_idxs]

        max_width = 192  # [img.shape[1] for img in imgs]

        imgs_pad = []
        imgs_wids = []

        for img in imgs:
            img = 255 - img
            img_height, img_width = img.shape[0], img.shape[1]
            outImg = np.zeros((img_height, max_width), dtype='float32')
            outImg[:, :img_width] = img[:, :max_width]

            img = 255 - outImg

            imgs_pad.append(self.transform(Image.fromarray(img.astype(np.uint8))))
            imgs_wids.append(img_width)

        imgs_pad = torch.cat(imgs_pad, 0)

        item = {
            'simg': imgs_pad,  # widths of the N images [list(N)]
            'swids': imgs_wids,  # N images (15) that come from the same author [N (15), H (32), MAX_W (192)]
            'img': real_img,  # the input image [1, H (32), W]
            'label': real_labels,  # the label of the input image [byte]
            'img_path': 'img_path',
            'idx': sample['img_id'],
            'wcl': int(author_id)  # id of the author [int]
        }
        return item

class FidDatasetNum(FidDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # from collections import Counter
        # import csv
        # labels = [d['label'] for d in sum(self.IMG_DATA.values(), [])]
        # counter = Counter(''.join(labels))
        # alphabet = ''.join(sorted(set(''.join(labels))))
        # with open('chars_count.csv', 'w', newline='') as f:
        #     writer = csv.writer(f)
        #     for char in alphabet:
        #         words_count = len([l for l in labels if char in l])
        #         writer.writerow([char, counter[char], words_count, words_count / len(labels)])

        self.IMG_DATA_OLD = self.IMG_DATA
        self.IMG_DATA = {wid: [s for s in samples if s['label'].isnumeric()] for wid, samples in self.IMG_DATA.items()}
        self.IMG_DATA = {wid: samples for wid, samples in self.IMG_DATA.items() if len(samples) > 0}
        self.alphabet = ''.join(sorted(set(''.join(d['label'] for d in sum(self.IMG_DATA.values(), [])))))
        self.author_id = sorted(self.IMG_DATA.keys())
        self.dataset_size = sum(len(samples) for samples in self.IMG_DATA.values())

    def __getitem__(self, index):
        NUM_SAMPLES = self.NUM_EXAMPLES
        sample, author_id = None, None
        for author_id, samples in self.IMG_DATA.items():
            if index < len(samples):
                sample, author_id = samples[index], author_id
                break
            index -= len(samples)

        self.IMG_DATA_AUTHOR = self.IMG_DATA_OLD[author_id]
        random_idxs = np.random.choice(len(self.IMG_DATA_AUTHOR), NUM_SAMPLES, replace=True)

        real_img = self.transform(sample['img'].convert('L'))
        real_labels = sample['label'].encode()

        imgs = [np.array(self.IMG_DATA_AUTHOR[idx]['img'].convert('L')) for idx in random_idxs]
        labels = [self.IMG_DATA_AUTHOR[idx]['label'].encode() for idx in random_idxs]

        max_width = 192  # [img.shape[1] for img in imgs]

        imgs_pad = []
        imgs_wids = []

        for img in imgs:
            img = 255 - img
            img_height, img_width = img.shape[0], img.shape[1]
            outImg = np.zeros((img_height, max_width), dtype='float32')
            outImg[:, :img_width] = img[:, :max_width]

            img = 255 - outImg

            imgs_pad.append(self.transform(Image.fromarray(img.astype(np.uint8))))
            imgs_wids.append(img_width)

        imgs_pad = torch.cat(imgs_pad, 0)

        item = {
            'simg': imgs_pad,  # widths of the N images [list(N)]
            'swids': imgs_wids,  # N images (15) that come from the same author [N (15), H (32), MAX_W (192)]
            'img': real_img,  # the input image [1, H (32), W]
            'label': real_labels,  # the label of the input image [byte]
            'img_path': 'img_path',
            'idx': sample['img_id'],
            'wcl': int(author_id)  # id of the author [int]
        }
        return item


class FidDatasetLong(FidDatasetNum):
    def __init__(self, *args, threshold=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.chars_freq = {"e": 33548, "t": 23659, "a": 21117, "o": 19330, "n": 18563, "i": 18108, "s": 16496, "r": 16427, "h": 14376, "l": 10598, "d": 10154, "c": 6958, "u": 6831, "m": 6286, "f": 5715, "w": 4860, "y": 4859, "g": 4840, "p": 4815, "b": 3962, ".": 3830, ",": 3098, "v": 2652, "k": 1614, "\"": 980, "-": 961, "'": 953, "M": 925, "T": 921, "I": 813, "A": 784, "S": 660, "B": 562, "P": 556, "H": 549, "W": 513, "C": 498, "N": 493, "G": 443, "R": 435, "x": 409, "L": 390, "E": 375, "D": 335, "F": 325, "0": 316, "1": 281, "j": 276, "O": 222, "q": 216, "U": 132, "K": 125, "(": 114, "3": 113, "?": 113, "9": 112, ")": 111, "z": 109, "2": 107, "J": 105, "V": 105, ":": 104, "Y": 104, ";": 100, "5": 97, "!": 91, "8": 86, "4": 75, "6": 70, "#": 55, " ": 43, "&": 38, "7": 36, "/": 16, "*": 9, "Q": 8, "X": 8, "Z": 5, "+": 2}
        long_tain = set(c for c, freq in self.chars_freq.items() if freq < threshold)
        self.IMG_DATA = {wid: [s for s in samples if len(set(s['label']) & long_tain)] for wid, samples in self.IMG_DATA_OLD.items()}
        self.IMG_DATA = {wid: samples for wid, samples in self.IMG_DATA.items() if len(samples) > 0}
        self.alphabet = ''.join(sorted(set(''.join(d['label'] for d in sum(self.IMG_DATA.values(), [])))))
        self.author_id = sorted(self.IMG_DATA.keys())
        self.dataset_size = sum(len(samples) for samples in self.IMG_DATA.values())

class FidDatasetRand(FidDataset):
    def __init__(self, rand_words_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(rand_words_path, 'r') as f:
            self.rand_words = f.read().splitlines()

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['label'] = random.choice(self.rand_words).encode('utf-8')
        return item

class FidDatasetApollo(FidDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['label'] = 'Apollo11'.encode('utf-8')
        return item


class FidDatasetAllNum(FidDataset):
    def __len__(self):
        return 10000

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['label'] = f'{index:04d}'.encode('utf-8')
        return item


class TextSentence(TextDataset):
    def __getitem__(self, index):
        from copy import deepcopy
        sentence = "The eagle has landed".split()
        item = super().__getitem__(index)
        res = []
        for i, word in enumerate(sentence):
            data = deepcopy(item)
            data['label'] = word.encode('utf-8')
            data['img'] = data['simg'][i % self.NUM_EXAMPLES].unsqueeze(0)
            res.append(data)
        return res

class TextSentenceval(TextDatasetval):
    def __getitem__(self, index):
        from copy import deepcopy
        sentence = "The eagle has landed".split()
        item = super().__getitem__(index)
        res = []
        for i, word in enumerate(sentence):
            data = deepcopy(item)
            data['label'] = word.encode('utf-8')
            data['img'] = data['simg'][i % self.NUM_EXAMPLES].unsqueeze(0)
            res.append(data)
        return res

class FolderDataset:
    def __init__(self, folder_path, num_examples=15):
        folder_path = Path(folder_path)
        self.imgs = list(folder_path.iterdir())
        self.transform = get_transform(grayscale=True)
        self.num_examples = num_examples

    def __len__(self):
        return len(self.imgs)

    def sample_style(self):
        random_idxs = np.random.choice(len(self.imgs), self.num_examples, replace=False)
        imgs = [Image.open(self.imgs[idx]).convert('L') for idx in random_idxs]
        imgs = [img.resize((img.size[0] * 32 // img.size[1], 32), Image.BILINEAR) for img in imgs]
        imgs = [np.array(img) for img in imgs]

        max_width = 192  # [img.shape[1] for img in imgs]

        imgs_pad = []
        imgs_wids = []

        for img in imgs:
            img = 255 - img
            img_height, img_width = img.shape[0], img.shape[1]
            outImg = np.zeros((img_height, max_width), dtype='float32')
            outImg[:, :img_width] = img[:, :max_width]

            img = 255 - outImg

            imgs_pad.append(self.transform(Image.fromarray(img.astype(np.uint8))))
            imgs_wids.append(img_width)

        imgs_pad = torch.cat(imgs_pad, 0)

        item = {
            'simg': imgs_pad,  # widths of the N images [list(N)]
            'swids': imgs_wids,  # N images (15) that come from the same author [N (15), H (32), MAX_W (192)]
        }
        return item