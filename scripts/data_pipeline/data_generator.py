import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
from utils import img_utils as img_util

logger = img_util.create_logger()

try:
    # Add Augmentation
    import albumentations as A
except ImportError as err:
    logger.error("{} pip install albumentations".format(err))




class DataGenerator(Sequence):
    def __init__(self, image_files, mask_files, batch_size=1, image_size=512, shuffle=True, transform=False):
        """Initialization
        :param image_files: list of all images ids (file names)
        :param mask_files: list of mask labels (file names)
        :param batch_size: create batch of images
        :param image_size: path to masks location
        :param shuffle: True to shuffle label indexes after every epoch
        :param augment: To set augmentation Flag True/False
        """
        self.image_filenames = image_files
        self.mask_names = mask_files
        self.batch_size = batch_size
        self.image_shape = (image_size, image_size)
        self.list_IDs = np.arange(len(self.image_filenames))
        self.shuffle = shuffle
        if transform:
            self.augmentation = self.augment_data()
        else:
            self.augmentation = None
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / (self.batch_size)))

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.image_filenames))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def read_image_mask(self, image_name, mask_name):
        input_img = img_util.read_image_mask(image_name, mask=False)
        input_img = cv2.resize(input_img, self.image_shape, cv2.INTER_AREA)
        input_img = input_img.astype(np.float32)

        mask_img = img_util.read_image_mask(mask_name, mask=True)

        mask_img = cv2.resize(mask_img, self.image_shape, cv2.INTER_AREA)
        mask_img = np.expand_dims(mask_img,axis=-1)
        mask_img = mask_img.astype(np.float32)

        return input_img / 255.0, mask_img / 255.0

    def transform_data(self, image, mask):
        transformed = self.augmentation(image=image, mask=mask)
        transform_image = transformed['image']
        transform_mask = transformed['mask']
        return transform_image, transform_mask

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Defining dataset
        X = np.empty((self.batch_size, self.image_shape[0], self.image_shape[1], 3), dtype=np.float32)
        Y = np.empty((self.batch_size, self.image_shape[0], self.image_shape[1], 1), dtype=np.float32)
        for i, index in enumerate(list_IDs_temp):
            X_sample, Y_sample = self.read_image_mask(self.image_filenames[index], self.mask_names[index])
            if self.augmentation is not None:
                X_sample, Y_sample = self.transform_data(X_sample, Y_sample)
            if self.batch_size == 1:
                X = np.reshape(X_sample, (1, X_sample.shape[0], X_sample.shape[1], 3))
                Y = np.reshape(Y_sample, (1, Y_sample.shape[0], Y_sample.shape[1], 1))
                return X, Y
            else:
                X[i, ...] = X_sample
                Y[i, ...] = Y_sample
        return X, Y

    @staticmethod
    def augment_data():
        return A.Compose([A.Transpose(p=0.5),
                          A.Rotate(p=0.2, limit=20),
                          A.HorizontalFlip(p=0.5),
                          A.VerticalFlip(p=0.5),
                          A.OneOf([A.RandomBrightnessContrast(p=0.2)], p=0.5)
                          ], p=1)


