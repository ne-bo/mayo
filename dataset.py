import datetime
import os.path
import pickle as pkl

import PIL
import numpy as np
from PIL.Image import Image
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from openslide import OpenSlide


class MayoDataset(Dataset):
    def __init__(self, test_or_train, train_batch_size=12):
        super().__init__()
        self.data_dir = "/home/natasha/Downloads/mayo/"
        # self.data_dir = "/scratch/natalia.pavlovskaia/mayo/"
        labels_dict = {'CE': 0, 'LAA': 1, 'Other': 2, 'Unknown': 3}
        if test_or_train in ['train', 'validation']:
            print('read train csv')
            data = pd.read_csv(self.data_dir + 'train.csv')[:17]
            print('data ', len(data))
            # print('df ', data)

            feature_names = list(data.columns.values)[1:]
            print('feature_names ', feature_names)
            dataset = []
            for idx, content in data.iterrows():
                if idx % 100 == 0:
                    print('idx', idx, datetime.datetime.now())
                dataset.append({'image_id': content['image_id'],
                                'center_id': content['center_id'],
                                'patient_id': content['patient_id'],
                                'image_num': int(content['image_num']),
                                'label': labels_dict[content['label']]})

            train_indices, val_indices = self.stratify_folds(dataset)

            if test_or_train in ['train', 'validation']:
                self.dataset = [el for idx, el in enumerate(dataset) if idx in train_indices]
                print('train is ok')
                self.dataset = np.array(self.dataset)
                print('self.dataset ', len(self.dataset))
                np.save('train.npy', self.dataset)

                self.dataset = [el for idx, el in enumerate(dataset) if idx in val_indices]
                print('val is ok')
                self.dataset = np.array(self.dataset)
                print('self.dataset ', len(self.dataset))

        if test_or_train == 'test':
            print('read test csv')
            data = pd.read_csv(self.data_dir + 'test.csv')  # [:500]
            print('data ', len(data))
            # print('df ', data)

            feature_names = list(data.columns.values)[1:]
            print('feature_names ', feature_names)
            dataset = []
            for idx, content in data.iterrows():
                if idx % 100 == 0:
                    print('idx', idx, datetime.datetime.now())
                dataset.append({'image_id': content['image_id'],
                                'center_id': content['center_id'],
                                'patient_id': content['patient_id'],
                                'image_num': int(content['image_num'])})
            self.dataset = [el for el in dataset]
            self.dataset = np.array(self.dataset)
            print('self.dataset ', len(self.dataset))
        # print('self.dataset ', self.dataset)
        print('self.dataset ', self.dataset)
        self.test_or_train = test_or_train

    def stratify_folds(self, dataset):
        skf = StratifiedKFold(n_splits=5)
        print('skf.split(dataset, [el[\'label\'] for el in dataset]) ',
              skf.split(dataset, [value['label'] for value in dataset]))
        for t, v in skf.split(dataset, [value['label'] for value in dataset]):
            train_indices = t
            val_indices = v
        print('val_indices ', val_indices, val_indices.shape)
        print('train_indices ', train_indices, train_indices.shape)
        return train_indices, val_indices

    def __getitem__(self, index):
        sample = self.dataset[index]
        tif_path = os.path.join(self.data_dir, 'train', sample['image_id'] + '.tif')
        npy_path = tif_path.replace('.tif', '.npy')
        png_path = tif_path.replace('.tif', '.png')
        if not os.path.exists(npy_path):
            slide = OpenSlide(tif_path)
            #print('sample', sample, slide.properties)
            downscale_factor = 100
            img = slide.read_region((0, 0), 0, slide.dimensions)\
                .resize((slide.dimensions[0]//downscale_factor, slide.dimensions[1]//downscale_factor))\
                .convert("RGB")
            img.save(png_path)
            #print('img ', img)
            img = np.array(img).transpose((2, 0, 1))
            print('img shape', img.shape)
            np.save(npy_path, img)
        else:
            img = np.load(npy_path)
        sample['image'] = img

        return sample

    def __len__(self):
        return len(self.dataset)


class DataModule(LightningDataModule):

    def __init__(
            self,
            train_batch_size: int = 200,
            eval_batch_size: int = 200,
            train_or_test='train',
            **kwargs,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.dataset = {}
        if train_or_test in ['train', 'validation']:
            self.dataset['train'] = MayoDataset('train')
            self.dataset['validation'] = MayoDataset('validation')
        else:
            print('create test dataset')
            self.dataset['test'] = MayoDataset('test')
            print('self.dataset[\'test\'] ', self.dataset['test'])
        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def train_dataloader(self):
        return DataLoader(self.dataset["train"],
                          batch_size=self.train_batch_size, shuffle=True
                          )

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
