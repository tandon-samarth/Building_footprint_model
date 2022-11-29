import argparse
import os
import warnings
from os import path as osp

from data_pipeline.data_generator import DataGenerator
from train_pipeline.segmentation_unet import TrainNetwork
from utils.img_utils import create_logger as logger

warnings.filterwarnings("ignore")

WORKDIR = os.getcwd()
outlog = logger()


def main(args):
    image_path = osp.join(args.path, 'Image')
    rgb_images = [osp.join(image_path, image_names) for image_names in os.listdir(image_path)]
    mask_images = [fname.replace('RGBImages', 'Mask') for fname in rgb_images]
    outlog.info('{} Images used for training'.format(len(rgb_images)))
    display_info(args)
    train_generator = DataGenerator(image_files=rgb_images,
                                    mask_files=mask_images,
                                    batch_size=args.batch_size,
                                    image_size=args.imgsize,
                                    transform=True
                                    )

    X_train, y_train = train_generator.__getitem__(0)
    outlog.info("Model Input data shape {}".format(X_train.shape))
    outlog.info("Model Output data shape {}".format(y_train.shape))

    train_model = TrainNetwork(backbone=args.backbone,
                               optimizer=args.optimizer,
                               n_classes=args.classes,
                               epochs=args.epochs,
                               lr=args.lr,
                               batch_size=args.batch_size,
                               loss=args.loss, metrics=args.metrics,
                               output_path=osp.join(args.path, args.out_path)
                               )

    # compile model
    train_model.transform(test_generator=train_generator,monitor_param='dice_metric')
    train_model.fit(train_generator)
    outlog.info("Model Training done..")
    return 0

def display_info(args):
    if args.verbose:
        outlog.info("Input Image size:{}".format(args.imgsize))
        outlog.info("N_classes:{}".format(args.classes))
        outlog.info("LossFunction:{}\n\tEpochs:{}\n\tBatch_size:{}".format(args.loss, args.epochs, args.batch_size))
        outlog.info("Backbone:{}\n\tMetrics:{}\n\tOptimizer:{}".format(args.backbone, args.metrics, args.optimizer))
        outlog.info("Model artifacts stored at {}".format(osp.join(args.path, args.out_path)))
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data and Train Pipeline for building footprint detection')
    parser.add_argument('-p', '--path', default=None, required=True,
                        help='Absolute Path to the directory having images of mask and rgb')
    parser.add_argument('--epochs', default=50, required=False, type=int, help='Set the number of epochs in Train')
    parser.add_argument('--loss', type=str, default='jaccard_distance',
                        help='loss function jaccard_distance/dice/bce_dice/bce_logdice')
    parser.add_argument('--classes', type=int, default=1, help='Number of classes default 1')
    parser.add_argument('--metrics', type=str, default='all', help="Metrics accuracy/iou/jaccard/dice default=all")
    parser.add_argument('--backbone', default='efficientnetb7', required=False, type=str, help='Model backbone')
    parser.add_argument('--optimizer', default='adam', type=str, help='Optimizer configration default Adam')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size for training default 8')
    parser.add_argument('--lr', type=float, default=0.001, help='Model learning rate')
    parser.add_argument('--imgsize', type=int, default=512, help='image size of the image')
    parser.add_argument('--verbose', type=int, default=0, help='show output 0/1')
    parser.add_argument('-o', '--out_path', default='runs', help='Path to save output artifacts')
    args = parser.parse_args()
    main(args)
