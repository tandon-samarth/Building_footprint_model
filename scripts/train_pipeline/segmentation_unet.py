import os

import cv2

os.environ["SM_FRAMEWORK"] = 'tf.keras'
from os import path as osp
import tensorflow as tf
from segmentation_models import Unet
from segmentation_models import get_preprocessing
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from utils import tf_utils as tf_util
from utils.img_utils import decode_segmentation_masks_gray

class TrainNetwork:
    def __init__(self, backbone='efficientnetb7', optimizer='adam', n_classes=1, epochs=50, batch_size=8,
                 lr=0.001, loss='bce_logdice', metrics='all', output_path=None):
        self.callbacks = None
        self.BACKBONE = backbone
        self.preprocess_input = get_preprocessing(self.BACKBONE)
        self.n_class = n_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.outpath = output_path
        # define save paths
        if self.outpath:
            os.makedirs(osp.join(self.outpath, 'models', self.BACKBONE), exist_ok=True)
            os.makedirs((osp.join(self.outpath, 'artifacts', self.BACKBONE)), exist_ok=True)
        # Define Optimizer
        if optimizer == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        elif optimizer == 'adagrad':
            self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=self.lr)
        # define Loss Functions
        if loss == 'bce_logdice':
            self.loss = tf_util.bce_logdice_loss
        elif loss == 'jaccard_distance':
            self.loss = tf_util.jaccard_distance_loss
        elif loss == 'dice':
            self.loss = tf_util.dice_loss
        elif loss == 'bce_dice':
            self.loss = tf_util.bce_dice_loss
        # define Accuracy Metrics
        if metrics == 'all':
            self.metrics = ['accuracy',
                            tf_util.jaccard_coef,
                            tf_util.jaccard_coef_int,
                            tf_util.dice_metric,
                            tf.keras.metrics.MeanIoU(num_classes=2)]
        elif metrics == 'iou':
            self.metrics = [tf.keras.metrics.MeanIoU(num_classes=2)]
        elif metrics == 'jaccard':
            self.metrics = [tf_util.jaccard_coef, tf_util.jaccard_coef_int, tf_util.dice_metric]

        # define Model backbone
        self.eff_b7 = Unet(self.BACKBONE, encoder_weights='imagenet', classes=self.n_class)

    def transform(self, test_generator=None, min_lr=0.0001, monitor_param='val_jaccard_coef_int',
                          patience=10):
        # Model Compile
        self.eff_b7.compile(optimizer=self.optimizer,
                            loss=tf_util.bce_logdice_loss,
                            metrics=self.metrics)
        # Define Callbacks
        reduce_lr = ReduceLROnPlateau(monitor=monitor_param, factor=0.5, mode='max', patience=patience, min_lr=min_lr)
        # stop learning as metric on validation stop increasing
        early_stopping = EarlyStopping(monitor=monitor_param, patience=patience, verbose=1, mode='max')

        class DisplayCallback(tf.keras.callbacks.Callback):
            def __init__(self, patience=10, save_path=None):
                super(DisplayCallback, self).__init__()
                self.patience = patience
                self.outpath = save_path

            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % self.patience == 0:
                    x_sample, y_sample = test_generator.__getitem__(np.random.randint(0, 10))
                    if x_sample.shape[0] > 1:
                        x_sample = x_sample[0]
                        input_img = np.expand_dims(x_sample, axis=0)
                    else:
                        input_img = x_sample
                    predicted_mask = self.model.predict(input_img)
                    predicted_mask = np.reshape(predicted_mask, (predicted_mask.shape[1], predicted_mask.shape[2], 1))
                    predicted_mask = decode_segmentation_masks_gray(predicted_mask)
                    cv2.imwrite(osp.join(self.outpath, 'epoch_' + str(epoch + 1) + '_sample.png'), predicted_mask)

        artifacts_dir = osp.join(self.outpath, 'artifacts')
        model_path = osp.join(self.outpath, 'models')
        checkpoint = ModelCheckpoint(model_path, monitor=monitor_param, verbose=1, save_best_only=True,
                                     mode='max')
        if test_generator:
            self.callbacks = [reduce_lr, early_stopping, DisplayCallback(save_path=artifacts_dir, patience=patience),
                          checkpoint]
        else:
            self.callbacks = [reduce_lr, early_stopping,checkpoint]

    def fit(self,train_generator,test_generator=None):
        # Model fit
        history = self.eff_b7.fit(train_generator, epochs=self.epochs, batch_size=self.batch_size, verbose=1,
                                  callbacks=self.callbacks)
        if test_generator:
            history = self.eff_b7.fit(train_generator,
                                      epochs=self.epochs, batch_size=self.batch_size, verbose=1,
                                      validation_data=test_generator,
                                      callbacks=self.callbacks)
        return history


