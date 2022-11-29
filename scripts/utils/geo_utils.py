import os
import subprocess
import logging
import numpy as np
from PIL import Image
try:
    from osgeo import gdal, ogr
except ImportError as err:
    logging.warning("{} osgeo not installed ".format(err))

def create_poly_mask(rasterSrc, vectorSrc, npDistFileName='', noDataValue=0, burn_values=1):
    '''
    Create polygon mask for rasterSrc,
    Similar to labeltools/createNPPixArray() in spacenet utilities
    '''
    ## open source vector file that truth data
    source_ds = ogr.Open(vectorSrc)
    source_layer = source_ds.GetLayer()

    ## extract data from src Raster File to be emulated
    ## open raster file that is to be emulated
    srcRas_ds = gdal.Open(rasterSrc)
    cols = srcRas_ds.RasterXSize
    rows = srcRas_ds.RasterYSize

    if npDistFileName == '':
        dstPath = ".tmp.tiff"
    else:
        dstPath = npDistFileName

    ## create First raster memory layer, units are pixels
    # Change output to geotiff instead of memory
    memdrv = gdal.GetDriverByName('GTiff')
    dst_ds = memdrv.Create(dstPath, cols, rows, 1, gdal.GDT_Byte,
                           options=['COMPRESS=LZW'])
    dst_ds.SetGeoTransform(srcRas_ds.GetGeoTransform())
    dst_ds.SetProjection(srcRas_ds.GetProjection())
    band = dst_ds.GetRasterBand(1)
    band.SetNoDataValue(noDataValue)
    gdal.RasterizeLayer(dst_ds, [1], source_layer, burn_values=[burn_values])
    dst_ds = 0
    mask_image = Image.open(dstPath)
    mask_image = np.array(mask_image)

    if npDistFileName == '':
        os.remove(dstPath)
    return mask_image


def convertTo8Bit(rasterImageName, outputRaster, outputPixType='Byte', outputFormat='GTiff', rescale_type='rescale',
                  percentiles=[2, 98]):
    '''
    This does a relatively poor job of converting to 8bit
    rescale_type = [clip, rescale]
        if resceale, each band is rescaled to its own min and max
        if clip, scaling is done sctricly between 0 65535
    '''

    srcRaster = gdal.Open(rasterImageName)
    nbands = srcRaster.RasterCount
    if nbands == 3:
        cmd = ['gdal_translate', '-ot', outputPixType, '-of', outputFormat,
               '-co', '"PHOTOMETRIC=rgb"']
    else:
        cmd = ['gdal_translate', '-ot', outputPixType, '-of', outputFormat]

    for bandId in range(srcRaster.RasterCount):
        bandId = bandId + 1
        band = srcRaster.GetRasterBand(bandId)
        if rescale_type == 'rescale':
            bmin = band.GetMinimum()
            bmax = band.GetMaximum()
            # if not exist minimum and maximum values
            if bmin is None or bmax is None:
                (bmin, bmax) = band.ComputeRasterMinMax(1)
            # else, rescale
            band_arr_tmp = band.ReadAsArray()
            bmin = np.percentile(band_arr_tmp.flatten(), percentiles[0])
            bmax = np.percentile(band_arr_tmp.flatten(), percentiles[1])

        else:
            bmin, bmax = 0, 65535

        cmd.append('-scale_{}'.format(bandId))
        cmd.append('{}'.format(bmin))
        cmd.append('{}'.format(bmax))
        cmd.append('{}'.format(0))
        cmd.append('{}'.format(255))

    cmd.append(rasterImageName)
    cmd.append(outputRaster)
    subprocess.call(cmd)
    return

