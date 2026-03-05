import os
import sys
import json
import time
import datetime
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
from cv2geojson import find_geocontours, export_annotations
from scipy.ndimage import gaussian_filter, median_filter

"""
    This version will attempt to make precise geojson outlines of as many tile 
    label types as are present in the label_col of associated tile dataframe list .tsv file
"""


def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


min_area = 0  # Microns square
tile_df_pn = Path(sys.argv[1])
output = Path(sys.argv[2])
label_col = sys.argv[3]
method = sys.argv[4]
output.mkdir(parents=True, exist_ok=True)
print("outputs to", output)
print("Loading", tile_df_pn)
tile = pd.read_csv(tile_df_pn, sep="\t")
# slide_name = Path(tile.loc[0,'cur_path']).parent.parts[-1]
# print(slide_name,tile.shape)
start = time.time()
# Extract annotation contours
plot = False
tile_size = int(tile.sz.values[0])
ds = 5
ds_ts = tile_size // ds
h_ds_ts = ds_ts // 2
offset = tile_size // 2
mx = int((np.max(tile.x) + offset) // ds) + 1
my = int((np.max(tile.y) + offset) // ds) + 1
slide_name = tile_df_pn.parts[-1].split("_")[0]
new_feats = []
for label in tqdm(tile.loc[:, label_col].unique()):
    roi = np.zeros((mx, my))
    idx = tile.loc[:, label_col] == label
    xx = (tile.loc[idx, "x"].astype(int).values + offset) // ds
    yy = (tile.loc[idx, "y"].astype(int).values + offset) // ds

    if method == "default":
        for x, y in zip(xx, yy):
            roi[
                (x - h_ds_ts + 1) : (x + h_ds_ts - 1),
                (y - h_ds_ts + 1) : (y + h_ds_ts - 1),
            ] = 255
        local_mean = gaussian_filter(roi, sigma=1)  # very limited smoothing
    elif method == "dots":
        roi[xx, yy] = 255
        local_mean = gaussian_filter(roi, sigma=9)  #
    roi = roi.T
    local_mean = gaussian_filter(roi, sigma=1)  # very limited smoothing
    im = cv2.normalize(
        src=local_mean,
        dst=None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8UC1,
    )
    _, im = cv2.threshold(im, 20, 255, cv2.THRESH_BINARY)
    geocontours = find_geocontours(im, mode="imagej")
    features = [
        contour.export_feature(color=(0, 255, 0), label=label)
        for contour in geocontours
    ]

    for feat in features:
        new_feat = feat
        coords = np.array(feat["geometry"]["coordinates"])
        if len(coords) > 1:
            multi_coord = []
            for ii in range(0, len(coords)):
                coord = ((np.array(coords[ii])) * ds).tolist()
                multi_coord.append(coord)
            new_feat["geometry"]["coordinates"] = multi_coord
        else:
            new_feat["geometry"]["coordinates"] = (coords * ds).tolist()
        new_feats.append(new_feat)

fn = output.joinpath(
    "%s_ds5_10sm_tile_offset_112px_%s_labeling.geojson" % (slide_name, label_col)
)
print(fn)
export_annotations(new_feats, str(fn))
stop = time.time()
dur = str(datetime.timedelta(seconds=(stop - start)))
print("Geojson Wall time = %s H:M:S" % dur)
