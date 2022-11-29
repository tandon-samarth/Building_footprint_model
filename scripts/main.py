import argparse
import os
import warnings
from os import path as osp
import azureml.core as azure_core
import time
from data_pipeline.data_generator import DataGenerator
from train_pipeline.segmentation_unet import TrainNetwork
from utils.img_utils import create_logger as logger
from utils.azutils import download_datasets

warnings.filterwarnings("ignore")

VM_BASE_PATH = "/home/"
outlog = logger()

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
parser.add_argument('--verbose', type=int, default=1, help='show output 0/1')
parser.add_argument('-o', '--out_path', default='runs', help='Path to save output artifacts')
args = parser.parse_args()

# Get the experiment run context
run = azure_core.Run.get_context()
# Get Azure Machine Learning workspace
ws = run.experiment.workspace

# Download all data at specified VM path
download_time_start = time.time()
download_datasets(ws, args.ground_truth_data, VM_BASE_PATH)
download_duration = round(time.time() - download_time_start)

run.log("Download dataset duration", f"{download_duration // 60}min {download_duration % 60}s")
run.log("os path in vm ".format(os.system('ls /home/')))

image_path = osp.join(VM_BASE_PATH, args.path, 'Image')
rgb_images = [osp.join(image_path, image_names) for image_names in os.listdir(image_path)]
mask_images = [fname.replace('RGBImages', 'Mask') for fname in rgb_images]
run.log('{} Images used for training'.format(len(rgb_images)))

if args.verbose:
    run.log("Input Image size:{}".format(args.imgsize))
    run.log("N_classes:{}".format(args.classes))
    run.log("LossFunction:{}\n\tEpochs:{}\n\tBatch_size:{}".format(args.loss, args.epochs, args.batch_size))
    run.log("Backbone:{}\n\tMetrics:{}\n\tOptimizer:{}".format(args.backbone, args.metrics, args.optimizer))
    run.log("Model artifacts stored at {}".format(osp.join(args.path, args.out_path)))

train_generator = DataGenerator(image_files=rgb_images,
                                mask_files=mask_images,
                                batch_size=args.batch_size,
                                image_size=args.imgsize,
                                transform=True
                                )

X_train, y_train = train_generator.__getitem__(0)
run.log("Model Input data shape {}".format(X_train.shape))
run.log("Model Output data shape {}".format(y_train.shape))

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
train_model.transform(test_generator=train_generator, monitor_param='dice_metric')
# train_model.fit(train_generator)
run.log("Model Training done..")
