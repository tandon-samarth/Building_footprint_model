import argparse
import os
import warnings
from os import path as osp
import azureml.core as azure_core
import time
from data_pipeline.data_generator import DataGenerator
from train_pipeline.segmentation_unet import TrainNetwork
from utils.img_utils import create_logger as logger
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

warnings.filterwarnings("ignore")
outlog = logger()
try :
    import mlflow
except ImportError as err:
    outlog.error("Error: {}".format(err))

# Enter details of your AzureML workspace
subscription_id = '2888fde7-9a5c-48fc-8623-84f525de174c'
resource_group = 'poi_datalake'
workspace_name = 'poi_machine_learning_workspace'

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

BASE_PATH = os.getcwd()

ml_client = MLClient(credential=DefaultAzureCredential(),
                     subscription_id=subscription_id,
                     resource_group_name=resource_group)

mlflow_tracking_uri = ml_client.workspaces.get(workspace_name).mlflow_tracking_uri

# Get the experiment run context
run = azure_core.Run.get_context()
# Get Azure Machine Learning workspace
ws = run.experiment.workspace
run.log("SubscriptionID:{}", f"{ws.subscription_id}s")

# # Download all data at specified VM path
# download_time_start = time.time()
# download_datasets(ws,args.path,VM_BASE_PATH)
# download_duration = round(time.time() - download_time_start)

# image_path = osp.join(BASE_PATH, args.path, 'Image')
# rgb_images = [osp.join(image_path, image_names) for image_names in os.listdir(image_path)]
# mask_images = [fname.replace('RGBImages', 'Mask') for fname in rgb_images]
# run.log('{} Images used for training', f"{len(rgb_images)}")


if args.verbose:
    run.log("Input Image size:{}", f"{args.imgsize}")
    run.log("N_classes:{}".format(args.classes))
    run.log("LossFunction:{}\n\tEpochs:{}\n\tBatch_size:{}".format(args.loss, args.epochs, args.batch_size))
    run.log("Backbone:{}\n\tMetrics:{}\n\tOptimizer:{}".format(args.backbone, args.metrics, args.optimizer))
    run.log("Model artifacts stored at {}".format(osp.join(args.path, args.out_path)))

# train_generator = DataGenerator(image_files=rgb_images,
#                                 mask_files=mask_images,
#                                 batch_size=args.batch_size,
#                                 image_size=args.imgsize,
#                                 transform=True
#                                 )
#
# X_train, y_train = train_generator.__getitem__(0)
# run.log("Model Input data shape {}".format(X_train.shape))
# run.log("Model Output data shape {}".format(y_train.shape))
#
# train_model = TrainNetwork(backbone=args.backbone,
#                            optimizer=args.optimizer,
#                            n_classes=args.classes,
#                            epochs=args.epochs,
#                            lr=args.lr,
#                            batch_size=args.batch_size,
#                            loss=args.loss, metrics=args.metrics,
#                            output_path=osp.join(args.path, args.out_path)
#                            )

# compile model
# train_model.transform(test_generator=train_generator, monitor_param='dice_metric')
# stime = time.time()
# train_model.fit(train_generator)
# duration = round(time.time() - stime)
# run.log("Model Training done in", f"{duration // 60}min {duration % 60}s")
