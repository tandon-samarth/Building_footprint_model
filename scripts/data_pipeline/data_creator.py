import os
import urllib
import geopandas as gpd
import cv2
import numpy as np
import rasterio
import mercantile
from tqdm import tqdm
from shapely.geometry import Polygon
from tenacity import retry, stop_after_attempt
from rasterio.transform import Affine
from rasterio import features as fet

class CreateData:
    def __init__(self, ground_truth_path, out_path='output', img_width=512, img_height=512, zoom_level=17,
                 prefix='County'):
        self.__img_width__ = img_width
        self.__img_height__ = img_height
        self.__zoom__ = zoom_level

        self.gt_mask = gpd.read_file(ground_truth_path, crs=4326)
        self.spartial_index = self.gt_mask.sindex
        self.out_path = out_path
        self.prefix = prefix

    def create_data_for_county(self, geojson_path, crs='4326'):
        gdf = gpd.read_file(geojson_path, crs=4326)
        bounds = gdf.geometry.loc[0].bounds

        tiles = mercantile.tiles(bounds[0], bounds[1], bounds[2], bounds[3], zooms=self.__zoom__)

        for i, tile in tqdm(enumerate(tiles)):
            val = mercantile.bounds(tile)

            img = self.extractSatelliteImages(val.west, val.south, val.east, val.north,
                                              height=str(self.__img_height__),
                                              width=str(self.__img_width__),
                                              )

            polygon = Polygon(
                [(val.west, val.south), (val.west, val.north), (val.east, val.north), (val.east, val.south),
                 (val.west, val.south)])
            poly_gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs=self.gt_mask.crs)
            possible_matches_index = list(self.spartial_index.intersection(polygon.bounds))
            possible_matches = self.gt_mask.iloc[possible_matches_index]
            bounds = val.west, val.south, val.east, val.north

            if possible_matches.empty:
                img = np.zeros([self.__img_height__, self.__img_width__, 1], dtype=np.uint8)
                continue
            else:
                self.convertToTiff(img, val.south, val.west, val.north, val.east,
                                   height=self.__img_height__,
                                   width=self.__img_width__,
                                   path=os.path.join(self.out_path, 'image'),
                                   imagename='image_{}_{}_{}_{}'.format(self.__img_height__, self.__zoom__, self.prefix,
                                                                        i)
                                   )
                mask = self.vector_to_raster(possible_matches.geometry, self.bbox_to_affine(bounds, img_shape=(
                    self.__img_height__, self.__img_width__)),
                                             img_shape=(self.__img_height__, self.__img_width__))
                mask = np.expand_dims(mask, axis=-1)
            mask = mask * 255
            self.convertToTiff(mask, val.south, val.west, val.north, val.east,
                               height=self.__img_height__,
                               width=self.__img_width__,
                               path=os.path.join(self.out_path, 'mask'),
                               chanenls=1,
                               imagename='mask_{}_{}_{}_{}'.format(self.__img_height__, self.__zoom__, self.prefix, i))
        return 0

    @staticmethod
    @retry(stop=stop_after_attempt(3))
    def extractSatelliteImages(minX, minY, maxX, maxY, height='512', width='512'):
        url = f"http://wms3.mapsavvy.com/WMSService.svc/db45ac1c32ac4e9caa5ecc3473998c81/WMSLatLon?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS=Aerial&SRS=EPSG:4326&CRS=EPSG:4326&BBOX={minX},{minY},{maxX},{maxY}&WIDTH={width}&HEIGHT={height}&STYLES=&TRANSPARENT=false&FORMAT=image/png"
        req = urllib.request.Request(url)
        resp = urllib.request.urlopen(req)
        arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def convertToTiff(img, minlat, minlong, maxlat, maxlong, height=512, width=512, imagename='sample_image',
                      path='output', crs="epsg:4326", chanenls=3, prefix=1):
        if not os.path.isdir(path):
            os.makedirs(path)

        filename = imagename + '.tiff'
        if os.path.isfile(filename):
            return
        transform = rasterio.transform.from_bounds(minlong, minlat, maxlong, maxlat, width, height)
        with rasterio.open(os.path.join(path, filename), 'w', driver='GTiff', dtype=rasterio.uint8, count=3,
                           width=width, height=height, transform=transform, crs=crs) as dst:
            for index in range(chanenls):
                dst.write(img[:, :, index], indexes=index + 1)

    @staticmethod
    def extract_mask_images(df, minlat, minlong, maxlat, maxlong, spatial_index, width, height,
                            image_name='sample_mask',
                            out_path='output_mask'):
        polygon = Polygon(
            [(minlong, minlat), (minlong, maxlat), (maxlong, maxlat), (maxlong, minlat), (minlong, minlat)])
        poly_gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs="epsg:4326")
        possible_matches_index = list(spatial_index.intersection(polygon.bounds))
        possible_matches = df.iloc[possible_matches_index]
        bounds = minlong, minlat, maxlong, maxlat
        return possible_matches , bounds

    @staticmethod
    def vector_to_raster(road_geom, src_transform, img_shape=(512, 512)):
        return fet.rasterize(
            [shape for shape in road_geom],
            out_shape=img_shape,
            transform=src_transform,
            all_touched=True,
            fill=0,
            default_value=1,
            dtype=rasterio.uint8,
        )

    @staticmethod
    def bbox_to_affine(bounds, img_shape: tuple):
        xmin, ymin, xmax, ymax = bounds
        width, height = img_shape
        xres = (xmax - xmin) / float(width)
        yres = (ymax - ymin) / float(height)
        return Affine(xres, 0, xmin, 0, -yres, ymax)
