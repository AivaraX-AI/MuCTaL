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
from scipy.ndimage import gaussian_filter


def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


min_area = 100000  # Microns square
tile_df_pn = Path(sys.argv[1])
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
print("outputs to", output)
print("Loading", tile_df_pn)
tile = pd.read_csv(tile_df_pn, sep="\t")
slide_name = Path(tile.loc[0, "cur_path"]).parent.parts[-1]
print(slide_name, tile.shape)
start = time.time()
# Extract annotation contours
plot = False
p_thresh = 0.5  # 0.5 is default / should be best on avereage
tile_size = int(tile.sz.values[0])
ds = 5
offset = tile_size // 2
mx = int((np.max(tile.x) + offset) // ds) + 1
my = int((np.max(tile.y) + offset) // ds) + 1
tum = np.zeros((mx, my))
ntum = np.zeros((mx, my))
idx = tile.p_pos.values > p_thresh
xx = (tile.loc[idx, "x"].astype(int).values + offset) // ds
yy = (tile.loc[idx, "y"].astype(int).values + offset) // ds
tum[xx, yy] = 1
tum = tum.T
local_mean = gaussian_filter(tum, sigma=20)  # 100/ds)
im = cv2.normalize(
    src=local_mean,
    dst=None,
    alpha=0,
    beta=255,
    norm_type=cv2.NORM_MINMAX,
    dtype=cv2.CV_8UC1,
)
_, im = cv2.threshold(im, 20, 255, 0, cv2.THRESH_BINARY)
geocontours = find_geocontours(im, mode="imagej")
features = [
    contour.export_feature(color=(0, 255, 0), label="Tumor") for contour in geocontours
]
new_feats = []
for feat in tqdm(features):
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

# Filter small detections out?
keep_feat = []
for feat in new_feats:
    coords = np.array(feat["geometry"]["coordinates"])  # Pixels
    area = 0
    if len(coords) > 1:
        multi_area = []
        for ii in range(0, len(coords)):
            coord = np.array(coords[ii]) * 0.2517
            multi_area.append(PolyArea(coord[:, 0], coord[:, 1]))
        area = multi_area[0] - np.sum(np.array(multi_area)[1:])  # area minus holes
    else:
        coords = np.squeeze(coords) * 0.2517
        area = PolyArea(coords[:, 0], coords[:, 1])
    if area > min_area:
        keep_feat.append(feat)

fn = output.joinpath("%s_ds5_10sm_tile_offset_112px_p0.5.geojson" % slide_name)
print(fn)
export_annotations(keep_feat, str(fn))
stop = time.time()
dur = str(datetime.timedelta(seconds=(stop - start)))
print("Geojson Wall time = %s H:M:S" % dur)
