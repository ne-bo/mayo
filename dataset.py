import datetime
import gc
import os.path

import PIL
import cv2
import numpy as np
import pandas as pd
from PIL.Image import Image
from albumentations import CoarseDropout, Compose, Flip, RandomRotate90, Resize
from openslide import OpenSlide
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset


class MayoDataset(Dataset):
    def __init__(self, test_or_train, train_batch_size=12):
        super().__init__()
        self.data_dir = "/media/natasha/Data1/mayo_data/"
        # self.data_dir = "/scratch/natalia.pavlovskaia/mayo/"
        labels_dict = {'CE': 0, 'LAA': 1, 'Other': 2, 'Unknown': 3}
        if test_or_train in ['train', 'validation']:
            print('read train csv')
            data = pd.read_csv(self.data_dir + 'train.csv')  # [:17]
            print('data ', len(data))
            # print('df ', data)

            feature_names = list(data.columns.values)[1:]
            print('feature_names ', feature_names)
            dataset = []
            for idx, content in data.iterrows():
                if idx % 100 == 0:
                    print('idx', idx, datetime.datetime.now())
                tif_path = os.path.join(self.data_dir, 'train', content['image_id'] + '.tif')
                if os.path.exists(tif_path):
                    dataset.append({'image_id': content['image_id'],
                                    'center_id': content['center_id'],
                                    'patient_id': content['patient_id'],
                                    'image_num': int(content['image_num']),
                                    'label': labels_dict[content['label']]})

            train_indices, val_indices = self.stratify_folds(dataset)

            if test_or_train in ['train']:
                self.dataset = [el for idx, el in enumerate(dataset) if idx in train_indices]
                print('train is ok')
                self.dataset = np.array(self.dataset)
                print('self.dataset ', len(self.dataset))
                # np.save('train.npy', self.dataset)
            if test_or_train in ['validation']:
                self.dataset = [el for idx, el in enumerate(dataset) if idx in val_indices]
                print('val is ok')
                self.dataset = np.array(self.dataset)
                print('self.dataset ', len(self.dataset))

        if test_or_train == 'test':
            print('read test csv')
            data = pd.read_csv(self.data_dir + 'test.csv')
            print('data ', len(data))
            # print('df ', data)

            feature_names = list(data.columns.values)[1:]
            print('feature_names ', feature_names)
            dataset = []
            for idx, content in data.iterrows():
                tif_path = os.path.join(self.data_dir, 'train', content['image_id'] + '.tif')
                if os.path.exists(tif_path):
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

    def get_iou(self, bb1, bb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters
        ----------
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner

        Returns
        -------
        float
            in [0, 1]
        """
        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] < bb1['y2']
        assert bb2['x1'] < bb2['x2']
        assert bb2['y1'] < bb2['y2']

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou

    def is_intersect(self, bbox1, bbox2):
        if bbox1[0] < bbox2[0] < bbox1[0] + bbox1[2] or \
                bbox1[1] < bbox2[1] < bbox1[1] + bbox1[3] or \
                bbox2[0] < bbox1[0] < bbox2[0] + bbox2[2] or \
                bbox2[1] < bbox1[1] < bbox2[1] + bbox2[3]:
            return self.get_iou({'x1': bbox1[0], 'y1': bbox1[1], 'x2': bbox1[0] + bbox1[2], 'y2': bbox1[1] + bbox1[3]},
                                {'x1': bbox2[0], 'y1': bbox2[1], 'x2': bbox2[0] + bbox2[2], 'y2': bbox2[1] + bbox2[3]})
        else:
            return 0.0

    def filter_bboxes(self, sorted_bboxex):
        filtered_bboxes = {}
        intersection_count = 0
        print('len(sorted_bboxex)', len(sorted_bboxex))
        for bbox1, size1 in sorted_bboxex.items():
            print('bbox1, size1 ', bbox1, size1, end=' ')
            current_box_intersection_count = 0
            for bbox2, size2 in sorted_bboxex.items():
                # print('bbox1, bbox2 ', bbox1, bbox2, ' size1,size2',size1, size2, self.is_intersect(bbox1, bbox2))

                if self.is_intersect(bbox1, bbox2) > 0.2 and bbox1 != bbox2:
                    # print('intersect')
                    intersection_count += 1
                    current_box_intersection_count += 1
                    if size1 > size2:
                        filtered_bboxes[bbox1] = size1
            print('current_box_intersection_count ', current_box_intersection_count, end=' ')
            if current_box_intersection_count == 0:
                filtered_bboxes[bbox1] = size1
            print('len(filtered_bboxes)', len(filtered_bboxes))
            # input()
        return intersection_count, filtered_bboxes

    def thresh_callback(self, src_gray):
        # compute the median of the single channel pixel intensities
        v = src_gray.mean()  # np.median(src_gray)
        print(src_gray, src_gray.mean(), src_gray.var())

        # apply automatic Canny edge detection using the computed median
        sigma = 0.33
        lower = v  # int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        print('lower ', lower, ' upper ', upper, 'v ', v)

        T, thresh_img = cv2.threshold(src_gray, lower, upper,
                                      cv2.THRESH_BINARY)

        img = PIL.Image.fromarray(thresh_img.astype('uint8'))
        # print('img shape', img.shape)
        img.save('thresh.png')

        canny_output = cv2.Canny(src_gray, lower, upper)

        contours, _ = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        print('len(contours) ', len(contours))
        src_gray_with_contours = cv2.drawContours(src_gray, contours, -1, (0, 255, 0), 3)
        PIL.Image.fromarray(src_gray_with_contours.astype('uint8')).save('with_contours.png')
        contours_poly = [None] * len(contours)
        bound_rect = [None] * len(contours)
        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 0, True)
            bound_rect[i] = cv2.boundingRect(contours_poly[i])
        sorted_bboxex = {}
        for rect in bound_rect:
            sorted_bboxex[rect] = rect[2] * rect[3]
        intersection_count = 100
        filtered_bboxes = sorted_bboxex
        # while intersection_count > 0:
        #     print('intersection_count ', intersection_count)
        #     input()
        #     intersection_count, filtered_bboxes = self.filter_bboxes(filtered_bboxes)

        sorted_bboxex = dict(sorted(filtered_bboxes.items(), key=lambda item: item[1]))

        return [bb for bb, size in sorted_bboxex.items()][-20:]

    def __getitem__(self, index):
        sample = self.dataset[index]
        tif_path = os.path.join(self.data_dir, 'train', sample['image_id'] + '.tif')
        npy_path = tif_path.replace('.tif', '.npy')
        png_path = tif_path.replace('.tif', '.png')
        if not os.path.exists(npy_path):
            try:
                slide = OpenSlide(tif_path)
                # print('sample', sample, slide.properties)
                downscale_factor = 100
                resized_img = slide.read_region((0, 0), 0, slide.dimensions) \
                    .resize((slide.dimensions[0] // downscale_factor, slide.dimensions[1] // downscale_factor))

                resized_img.convert("RGB").save(png_path)
                # print('img ', img)

                np.save(npy_path, np.array(resized_img.convert("RGB")).transpose((2, 0, 1)))

                # print('img shape', img.shape)
                print('os.path.exists(npy_path) ', os.path.exists(npy_path), npy_path)
                if os.path.exists(npy_path):
                    gray_img = np.array(resized_img.convert("L"))
                    bound_rect = self.thresh_callback(gray_img)
                    print('bound_rect ', bound_rect, len(bound_rect))
                    # input()
                    for idx, rect in enumerate(bound_rect):
                        if rect[2] > 1 and rect[3] > 1:
                            downscale_factor = 10
                            resized_box = slide.read_region((rect[0] * 100, rect[1] * 100), 0,
                                                            (rect[2] * 100, rect[3] * 100)) \
                                .resize(
                                (rect[2] * 100 // downscale_factor, rect[3] * 100 // downscale_factor)).convert(
                                "RGB")
                            resized_box.save(png_path.replace('.png', '_%d.png' % idx))
                            # print('img ', img)
                            resized_box = np.array(resized_box).transpose((2, 0, 1))
                            # print('resized_box shape', resized_box.shape)
                            np.save(npy_path.replace('.npy', '_%d.npy' % idx), resized_box)
                            resized_box = None
                            gc.collect()
                img = np.array(resized_img.convert("RGB")).transpose((2, 0, 1))
            except:
                if os.path.exists(tif_path):
                    print('problems with ', tif_path)
                    input()
                img = np.zeros((3, 77, 77))

        else:
            img = np.load(npy_path).transpose((1, 2, 0))

        if self.test_or_train == 'train':
            transform = Compose([
                Resize(height=256, width=256, always_apply=True),
                RandomRotate90(),
                CoarseDropout(),
                Flip()
            ])
        else:
            transform = Compose([
                Resize(height=256, width=256, always_apply=True)
            ])
        # print('img ',img.shape)
        augmented_image = transform(image=img)['image']
        # print('augmented_image ', augmented_image.shape)
        sample['image'] = augmented_image.transpose((2, 0, 1))
        # input()
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
            print('len(self.dataset[train] )', len(self.dataset['train']))
        else:
            print('create test dataset')
            self.dataset['test'] = MayoDataset('test')
            print('self.dataset[\'test\'] ', self.dataset['test'])
        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def train_dataloader(self):
        print('self.train_batch_size ', self.train_batch_size)
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
