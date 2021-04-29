import pandas as pd
import geopandas as gpd
from datacube import Datacube
import cv2
import numpy as np

import sys

sys.path.append("../scripts")
from dea_spatialtools import xr_rasterize


class Solution:
    """
    Base class for a solution
    Need to define a subclass with a mask() function
    """
    def __init__(self):
        # Load the datacube
        self.dc = Datacube(app="Getting started")

        # Load linescans
        self.linescan_datasets = self.dc.find_datasets(product="linescan")
        self.linescan_datasets = sorted(
            self.linescan_datasets, key=lambda ds: (ds.center_time, ds.id)
        )
        print(len(self.linescan_datasets), "linescan datasets available")

        # Load list of training linescans
        self.train_linescans = pd.read_csv(
            "resources/challenge1_train.csv", index_col="id"
        )
        print(len(self.train_linescans), "training linescan datasets available")

        # Load fire map polygons
        vector_file = "resources/fire_boundaries.shp"
        self.gdf = gpd.read_file(vector_file)
        self.gdf["SourceNameClean"] = self.gdf.apply(
            lambda row: self._clean_name(row.SourceName), axis=1
        )

        # Data caches
        self.linescan_cache = {}
        self.raster_cache = {}

    @staticmethod
    def _clean_name(name):
        """Make filenames match"""
        if name is None:
            res = None
        else:
            if name.upper()[-4::] == ".JPG":
                res = name.upper()[:-4].replace(" ", "_")
            else:
                res = name.upper().replace(" ", "_")
        return res

    def match_data(self):
        self.training_data = {}
        matched_polygons = set()
        for linescan_name in self.train_linescans.label:

            # Find the linescan
            linescan = next(
                ls
                for ls in self.linescan_datasets
                if ls.metadata_doc["label"] == linescan_name
            )

            # Find polygons that match this linescan
            matches = self.gdf[self.gdf.SourceNameClean == linescan_name]
            matched_polygons = matched_polygons | set(matches.index)

            if len(matches) > 0:
                self.training_data[linescan] = matches

        print(f"{len(self.training_data)} linescans with matching polygons")
        print(
            f"{100 * len(self.training_data) / len(self.train_linescans):.1f}% of linescans used"
        )
        print(f"{100 * len(matched_polygons) / len(self.gdf):.1f}% of polygons used")

    def cached_load(self, linescan_id):
        if linescan_id not in self.linescan_cache:
            src = self.dc.load(
                    product="linescan",
                    id=linescan_id,
                    output_crs="epsg:28355",
                    resolution=(-10, 10),
                )
            self.linescan_cache[linescan_id] = src
        return self.linescan_cache[linescan_id]

    def cached_rasterize(self, matches, src, linescan_id):
        if linescan_id not in self.raster_cache:
            self.raster_cache[linescan_id] = xr_rasterize(gdf=matches, da=src)

        return self.raster_cache[linescan_id]

    def error_fraction(self):
        """Fraction of incorrect points"""
        total_error = 0
        total_points = 0
        for ls in self.training_data:
            src = self.cached_load(ls.id)
            matches = self.training_data[ls]

            # Rasterise polygon
            target = self.cached_rasterize(matches, src, ls.id)

            # Number of errors
            error = target != self.mask(src.linescan)
            total_error += error.sum()
            total_points += error.shape[0] * error.shape[1]

        return float(total_error / total_points)

    def f1_score(self):
        """
        Score used by Challenge to evaluate solution
        https://en.wikipedia.org/wiki/F-score
        """
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for ls in self.training_data:
            src = self.cached_load(ls.id)
            matches = self.training_data[ls]

            # Rasterise polygon
            target = self.cached_rasterize(matches, src, ls.id)

            # Solution mask
            mask = self.mask(src.linescan)

            # True positives and negatives
            tp += int(np.sum(target * mask))
            tn += int(np.sum(target + mask == 0))

            # False positives and negatives
            fp += int(np.sum((target == 1) * (mask == 0)))
            fn += int(np.sum((target == 0) * (mask == 1)))

        if tp > 0:
            p = tp / (tp + fp)
            r = tp / (tp + fn)
        else:
            p = 0
            r = 0

        if p + r > 0:
            f1 = 2*p*r/(p+r)
        else:
            f1 = 0

        return f1

    def generate_submission(self, filename):
        """Generate the submission file using challenge data"""

        test = pd.read_csv("resources/challenge1_test.csv", index_col="id")

        fnames = test.label.unique()

        for file_stem in fnames:
            src = self.dc.load(
                product="linescan",
                label=file_stem,
                output_crs="epsg:28355",
                resolution=(-10, 10),
            )
            mask = self.mask(src.linescan)

            # iterate over the coordinates that are required for testing in the current linescan file
            for idx, ob in test.loc[test.label == file_stem].iterrows():
                result_tf = mask.sel(x=ob.x, y=ob.y, method="nearest").values[0]
                result_10 = int(result_tf == True)
                test.loc[
                    (test.label == file_stem) & (test.x == ob.x) & (test.y == ob.y),
                    "target",
                ] = result_10

        test.to_csv(filename, columns=["target"])


class Threshold(Solution):
    """
    Use a simple threshold to determine if a pixel is on fire
    """
    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_close_kernel(self, size):
        self.kernel_close = self._kernel(max(int(size),1))

    @staticmethod
    def _kernel(size):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        return kernel

    def mask(self, linescan):
        """Generate mask from linescan"""

        mask = linescan > self.threshold
        floatmask = np.array(mask, dtype='f8')[0,:,:]

        # Close holes
        mask_close = cv2.morphologyEx(floatmask, cv2.MORPH_CLOSE, self.kernel_close)

        mask[0,:,:] = mask_close > 0.0

        return mask


class AutoThreshold(Threshold):
    """
    Use the deviation from mean as a threshold
    """
    def set_threshold(self, threshold_std):
        self.threshold_std = threshold_std

    def mask(self, linescan):
        """Generate mask from linescan"""

        mean = linescan.mean()
        std = linescan.std()

        mask = linescan > mean + std * self.threshold_std
        floatmask = np.array(mask, dtype='f8')[0,:,:]

        # Close holes
        mask_close = cv2.morphologyEx(floatmask, cv2.MORPH_CLOSE, self.kernel_close)

        mask[0,:,:] = mask_close > 0.0

        return mask
