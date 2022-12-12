import geopandas as gpd
from shapely import geometry
import numpy as np
import urllib
import cv2
import rasterio
from tqdm import tqdm
import os
from multiprocessing import Pool
from tenacity import retry, stop_after_attempt


class CreateData:
    def __init__(self, geojson_path, out_path='output', img_width=512, img_height=512):
        self.geojson_path = geojson_path
        self.out_path = out_path
        self.buffer = 100
        self.img_width = img_width
        self.img_height = img_height

        grid_points = self.create_gridpoints()
        with Pool(8) as p:
            output = list(tqdm(p.imap(self.download_images, grid_points), total=len(grid_points)))

    def create_gridpoints(self):
        gdf = gpd.read_file(self.geojson_path, crs=4326).to_crs(3857)
        bounds = gdf.loc[0].geometry.bounds
        polygon = gdf.loc[0].geometry

        grid_points = []
        i, j = 0, 0
        for x in np.arange(bounds[0], bounds[2], self.buffer * 2):
            for y in np.arange(bounds[1], bounds[3], self.buffer * 2):
                point = geometry.Point(x, y)
                if polygon.intersects(point):
                    grid_points.append((i, j, point))
                    j += 1
            i += 1
            j = 0
        return grid_points

    def download_images(self, data):
        file_name = f"{data[0]}_{data[1]}"
        if not os.path.exists(os.path.join(self.out_path , file_name + '.tiff')):
            point = data[2]
            image = self.extractSatelliteImages(point.x, point.y, self.buffer, height='512', width='512')
            self.convertToTiff(image, point.y - self.buffer, point.x - self.buffer, point.y + self.buffer,
                               point.x + self.buffer, self.img_width, self.img_height,
                               file_name, self.out_path)
        return (data[0], data[1])

    @staticmethod
    @retry(stop=stop_after_attempt(3))
    def extractSatelliteImages(x, y, buffer=50, height='512', width='512'):
        maxX = x + buffer
        minX = x - buffer
        maxY = y + buffer
        minY = y - buffer

        url = f"http://wms3.mapsavvy.com/WMSService.svc/db45ac1c32ac4e9caa5ecc3473998c81/WMSLatLon?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS=Aerial&SRS=EPSG:3857&CRS=EPSG:3857&BBOX={minX},{minY},{maxX},{maxY}&WIDTH={width}&HEIGHT={height}&STYLES=&TRANSPARENT=false&FORMAT=image/png"
        req = urllib.request.Request(url)
        resp = urllib.request.urlopen(req)
        arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def convertToTiff(img, minlat, minlong, maxlat, maxlong, height, width, imagename, path, crs="epsg:3857"):
        filename = imagename + '.tiff'
        transform = rasterio.transform.from_bounds(minlong, minlat, maxlong, maxlat, width, height)
        with rasterio.open(os.path.join(path, filename), 'w', driver='GTiff', dtype=rasterio.uint8, count=3,
                           width=width, height=height, transform=transform, crs=crs) as dst:
            for index in range(3):
                dst.write(img[:, :, index], indexes=index + 1)

