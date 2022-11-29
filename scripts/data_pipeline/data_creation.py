import glob
import os
import os.path as osp
import numpy as np
import cv2
from utils import img_utils as img_util
from utils.geo_utils import convertTo8Bit, create_poly_mask
from sklearn.model_selection import train_test_split

logger = img_util.create_logger()

class DataCreation:
    def __init__(self, outputpixtype='Byte', out_format='GTiff', rescale_type='rescale', convert_to_rgb=False):
        self.outpixtype = outputpixtype
        self.out_format = out_format
        self.rescale_type = rescale_type
        self.covert_data_type = convert_to_rgb


    def create_rgb_masks(self, input_raster_path, geojson_path, outputdirectory='output_data', verbose=0):
        output_rgbpath = osp.join(outputdirectory, 'RGBImage')
        output_annotations = osp.join(outputdirectory, 'Mask')
        if not osp.isdir(output_rgbpath):
            os.makedirs(output_rgbpath, exist_ok=True)
        os.makedirs(output_annotations, exist_ok=True)

        listofRaster = sorted(glob.glob(os.path.join(input_raster_path, '*.tif')))
        listofgeojson = sorted(glob.glob(os.path.join(geojson_path, '*.geojson')))

        if verbose:
            logger.info("Total Raster images detected {}".format(len(listofRaster)))
            logger.info("Total Geo-jsons detected {}".format(len(listofgeojson)))

        if len(listofRaster) != len(listofgeojson):
            logger.error("images and Geo-jsons are a mismatch")

        for rasterImage, geoJson in zip(listofRaster, listofgeojson):
            if not osp.isfile(osp.join(output_rgbpath, osp.basename(rasterImage))):
                convertTo8Bit(rasterImageName=rasterImage,
                              outputRaster=osp.join(output_rgbpath, osp.basename(rasterImage)))
                mask = create_poly_mask(rasterImage, geoJson)
                mask_file = osp.join(output_annotations, osp.basename(rasterImage).split('.')[0] + '.png')
                cv2.imwrite(mask_file, mask)
        logger.info("Data created at {}".format(outputdirectory))

        rgb_images = sorted(glob.glob(output_rgbpath + '/*.tif'))
        mask_images = sorted(glob.glob(output_annotations + '/*.png'))
        return rgb_images, mask_images

    @staticmethod
    def split_dataset(rgb_images, mask_images, train_size=0.80, test_size=0.20):
        rgb_images = sorted(rgb_images)
        mask_images = sorted(mask_images)
        logger("split data in {} train and {} test".format(train_size, test_size))
        data = np.asarray([(rgb, mask) for rgb, mask in zip(rgb_images, mask_images)])
        x_train, x_test = train_test_split(data, train_size=train_size, test_size=test_size)
        logger.info('Total Training RGB images:{}'.format(x_train.shape[0]))
        logger.info('Total Test RGB images:{}'.format(x_test.shape[0]))
        return x_train, x_test
