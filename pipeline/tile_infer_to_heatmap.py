import os
import sys
import time
import datetime
import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from pathlib import Path
from pathml.preprocessing import (
    TissueDetectionHE,
    LabelWhiteSpaceHE,
    LabelArtifactTileHE,
)
from pathml.core import HESlide

tile_df_fn = Path(sys.argv[1])
slide_pn = Path(sys.argv[2])
output = Path(sys.argv[3])
df = pd.read_csv(tile_df_fn, sep="\t")
heatmap_path = output
heatmap_path.mkdir(parents=True, exist_ok=True)
base_file = slide_pn.parts[-1].split(".")[0]
dswsi_path = output.joinpath("ds_only")
dswsi_path.mkdir(parents=True, exist_ok=True)
nchunks = 50
ds = 10
thresh = 0.5
use_blur = False
s = time.time()
print("Finished")


class MplColorHelper:

    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)


# Generate colorbar:
w = 2
cmap = "RdYlBu"
COL = MplColorHelper(cmap, 0, 1)
val = np.arange(0, 1, 0.05)
im = np.zeros((len(val), w, 3))
bar = []
for i, v in enumerate(val):
    rgb = np.array(COL.get_rgb(1 - v)[0:-1]) * 255
    bar.append(rgb)
fig = plt.figure()
ax = fig.add_subplot()
plt.imshow(np.array(bar).reshape(len(val), 1, 3).astype("uint8"))
plt.xlabel("")
plt.yticks([0, len(val) / 2, len(val)])
ax.set_yticklabels([0, 0.5, 1])
ax.set_xticklabels("")
ax.invert_yaxis()
plt.ylabel("P(Pos)")
for a in "xy":
    plt.tick_params(
        axis=a,  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )
plt.box(False)
fns = [heatmap_path.joinpath("colorbar.eps"), heatmap_path.joinpath("colorbar.png")]
for fn in fns:
    plt.savefig(fn)


# make downsampled .png images from each WSI:
def ds_img_from_wsi(wsi_fn, nchunks, ds, verbose=False):
    blank_detect = LabelWhiteSpaceHE(
        label_name="blank",
        proportion_threshold=0.99,
    )
    art_detect = LabelArtifactTileHE(label_name="artifact")
    wsi = HESlide(str(wsi_fn))
    x, y = wsi.shape
    tile_size = y // nchunks
    tot_tiles = (x // tile_size) * (y // tile_size)
    print(x, y, tile_size, tot_tiles)
    print("original slide shape:", x, y)
    dsx = x // ds
    dsy = y // ds
    print("ds shape:", dsx, dsy)
    print("tile size:", tile_size, "total tiles:", tot_tiles)
    blank_image = np.zeros((dsx, dsy, 3), np.uint8) + 255
    blank_tot = 0
    img_tot = 0
    for i, tile in enumerate(wsi.generate_tiles(shape=tile_size, pad=False)):
        blank_detect.apply(tile)
        if tile.labels["blank"] == False:
            if verbose:
                print("Loading tile %d" % i)
            im = np.array(tile.image)
            img_tot += 1
            xx, yy = tile.coords
            imds = cv2.resize(
                im, (tile_size // ds, tile_size // ds), interpolation=cv2.INTER_CUBIC
            )
            dsxx = xx // ds
            dsyy = yy // ds

            blank_image[
                dsxx : (dsxx + imds.shape[1]), dsyy : (dsyy + imds.shape[0])
            ] = imds
        else:
            blank_tot += 1
            if verbose:
                print("Tile %d detected as blank... skipping" % i)
    print("%d images loaded, %d detected as blank" % (img_tot, blank_tot))
    return blank_image


save_fn = dswsi_path.joinpath("%s_full_ds%dx_min.png" % (base_file, ds))
if save_fn.exists() == False:
    print("processing:", save_fn)
    ds_svs = ds_img_from_wsi(slide_pn, nchunks, ds)
    ds_svs = cv2.cvtColor(ds_svs, cv2.COLOR_RGB2BGR)
    print("Save %s\n" % save_fn)
    cv2.imwrite(str(save_fn), ds_svs)
else:
    print(save_fn, "Already exists.. skipping.")

cmap = "RdYlBu"
COL = MplColorHelper(cmap, 0, 1)
bkg_fn = dswsi_path.joinpath("%s_full_ds%dx_min.png" % (base_file, ds))
if bkg_fn.exists():
    wsi = HESlide(str(slide_pn))
    x, y = wsi.shape
    print("original slide shape:", x, y)
    xds = x // ds
    yds = y // ds
    print("ds shape:", xds, yds)
    tile_fns = df.loc[:, "cur_path"].values
    use_p = df.loc[:, "p_pos"].values
    save_fn = heatmap_path.joinpath("%s_v10_tile_heatmap_ds%dx.png" % (base_file, ds))
    print("make:", save_fn)
    heat_map = np.zeros((xds, yds, 3), np.uint8) + 255
    for i in range(0, df.shape[0]):
        tx = int(df.loc[i, "x"])
        ty = int(df.loc[i, "y"])
        tile_size = int(df.loc[i, "sz"])
        ds_ts = tile_size // ds
        dsx = tx // ds
        dsy = ty // ds
        im = np.zeros((ds_ts, ds_ts, 3), np.uint8)
        p = use_p[i]
        rgb = np.array(COL.get_rgb(1 - p)[0:-1]) * 255
        rgb = rgb.reshape((1, 1, 3))
        im = im + rgb
        imds = cv2.resize(im, (ds_ts, ds_ts), interpolation=cv2.INTER_CUBIC)
        heat_map[dsy : (dsy + ds_ts), dsx : (dsx + ds_ts)] = imds
    heat_map = cv2.cvtColor(heat_map, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_fn), heat_map)
    bkg = cv2.imread(str(bkg_fn))
    bkg = cv2.cvtColor(cv2.cvtColor(bkg, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    comb = cv2.addWeighted(bkg, 0.6, heat_map, 0.4, 30)
    comb_fn = heatmap_path.joinpath(
        "%s_v10_pred_tumor_overlay_ds%dx.jpg" % (base_file, ds)
    )
    print(comb_fn)
    cv2.imwrite(str(comb_fn), comb)
    print("Saved!\n")
else:
    print(bkg_fn, " Not found!")
ss = time.time()
dur = str(datetime.timedelta(seconds=(ss - s)))
print("Heatmap Wall time = %s H:M:S" % dur)
print("Heatmap Finished")
