import os
import urllib
import geopandas as gpd
import cv2
import numpy as np
import rasterio
import mercantile
from tenacity import retry, stop_after_attempt


class CreateData:
    def __init__(self, out_path='output', img_width=512, img_height=512, zoom_level=17):
        self.__img_width__ = img_width
        self.__img_height__ = img_height
        self.__zoom__ = zoom_level

    def create_data_for_county(self, geojson_path, out_path='output', crs='4326', ):
        gdf = gpd.read_file(geojson_path, crs=4326)
        bounds = gdf.geometry.loc[0].bounds
        tiles = mercantile.tiles(bounds[0], bounds[1], bounds[2], bounds[3], zooms=self.__zoom__)
        for i, tile in enumerate(tiles):
            val = mercantile.bounds(tile)
            img = self.extractSatelliteImages(val.west, val.south, val.east, val.north,
                                              height=str(self.__img_height__),
                                              width=str(self.__img_width__),
                                              )
            self.convertToTiff(img, val.west, val.south, val.east, val.north,
                               height=self.__img_height__,
                               width=self.__img_width__,
                               path=out_path,
                               imagename='image_{}_{}_{}_{}'.format(self.__img_height__,
                                                                    self.__img_width__,
                                                                    self.__zoom__, i))

    @staticmethod
    @retry(stop=stop_after_attempt(3))
    def extractSatelliteImages(minX, minY, maxX, maxY, height='512', width='512'):
        url = f"http://wms3.mapsavvy.com/WMSService.svc/db45ac1c32ac4e9caa5ecc3473998c81/WMSLatLon?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS=Aerial&SRS=EPSG:3857&CRS=EPSG:3857&BBOX={minX},{minY},{maxX},{maxY}&WIDTH={width}&HEIGHT={height}&STYLES=&TRANSPARENT=false&FORMAT=image/png"
        req = urllib.request.Request(url)
        resp = urllib.request.urlopen(req)
        arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def convertToTiff(img, minlat, minlong, maxlat, maxlong, height=512, width=512, imagename='sample_image',
                      path='output', crs="epsg:4326"):
        filename = imagename + '.tiff'
        transform = rasterio.transform.from_bounds(minlong, minlat, maxlong, maxlat, width, height)
        with rasterio.open(os.path.join(path, filename), 'w', driver='GTiff', dtype=rasterio.uint8, count=3,
                           width=width, height=height, transform=transform, crs=crs) as dst:
            for index in range(3):
                dst.write(img[:, :, index], indexes=index + 1)
