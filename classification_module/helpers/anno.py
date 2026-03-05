import os
import glob
import json
import math
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from pathlib import Path


def plot_geojson_feature(
    feat, ax, class_cm={"Malignant": "r", "Benign Bile Duct": "b"}
):
    """feat = geojson qupath annotation feature
    ax = matplotlib axis object
    class_cm = dictionary mapping annotation classification names to plot colors
    plots: annotation polygon on axis
    """
    anno_class = feat["properties"]["classification"]["name"]
    anno_class_cm = [cm for cm in class_cm.keys()]
    if anno_class in anno_class_cm:
        col = class_cm[anno_class]
        for coord in feat["geometry"]["coordinates"]:
            xs, ys = zip(*coord)
            # Ensure closure:
            xs = np.append(xs, xs[0])
            ys = np.append(ys, ys[0])
            ax.plot(xs, ys, col, label=anno_class)
        # ax.invert_yaxis()
    return ax


def index_tiles_of_core_with_feature(feat, subset, tma_dat, rotate_180=False):
    """feat = geojson qupath annotation feature
    subset ->  dataframe of inferred tile names, preprocessed to subset ->   see: tiles_in_core_df_subset()
    returns: which tile feature is centered in (if any)
    """
    if rotate_180:
        core = subset.loc[0, "core"]
        core_width = tma_dat.loc[tma_dat.loc[:, "Core #"] == core, "Width"].values

    coords = np.squeeze(np.array(feat["geometry"]["coordinates"]))
    fx, fy = tuple(np.mean(coords, axis=0))
    if rotate_180:
        idx0 = (fx > (core_width - subset.norm_x)) & (
            fx <= (core_width - subset.norm_x + subset.width)
        )
        idx1 = (fy > (core_width - subset.norm_y)) & (
            fy <= (core_width - subset.norm_y + subset.width)
        )
    else:
        idx0 = (fx > subset.norm_x) & (fx <= (subset.norm_x + subset.width))
        idx1 = (fy > subset.norm_y) & (fy <= (subset.norm_y + subset.width))
    idx = idx0 & idx1
    return idx


def calc_percent_poscells_in_tumor_tiles(
    allobjects,
    subset,
    tma_dat,
    ax=None,
    use_plot=False,
    class_cm={"Positive": "yellow", "Negative": "cyan"},
    thresh=0.95,  # threhsold for considering tile in tumor
    rotate_180=False,
):
    """Take in a qupath geojson object of positive cell detections ('allobjects') e.g. IHC detections in one TMA core,
    then use dataframe of tile location and size, with prediction probabilities of tumor (subset df[:,'p_pos']),
    to return dictionary of number cells detected in- and outside tumor, positive for ihc (total_pos_in_tumor),
    or negative (total_in_tumor-total_pos_in_tumor) (output)
    not_in_tiles -> count indicates number of cells where cell location could not be mapped to a CNN tile; this can happen
        if tile was excluded for having too much white space, artifacts, other issues.
    """
    total = 0
    total_in_tumor = 0
    total_pos_in_tumor = 0
    total_pos_out_tumor = 0
    not_in_tiles = 0
    anno_class_cm = [cm for cm in class_cm.keys()]
    for feat in allobjects:
        anno_class = feat["properties"]["classification"]["name"]
        if anno_class in anno_class_cm:
            idx = index_tiles_of_core_with_feature(
                feat, subset, tma_dat, rotate_180=rotate_180
            )
            if any(idx):
                total = total + 1
                p_pos = subset.p_pos[idx] > thresh
                if any(p_pos):
                    total_in_tumor = total_in_tumor + 1
                    if anno_class == anno_class_cm[0]:  # assume this is positive
                        total_pos_in_tumor = total_pos_in_tumor + 1
                else:
                    if anno_class == anno_class_cm[0]:  # assume this is positive
                        total_pos_out_tumor = total_pos_out_tumor + 1
            else:
                not_in_tiles = not_in_tiles + 1
    if total_in_tumor > 0:
        percent = (total_pos_in_tumor / total_in_tumor) * 100
    else:
        percent = 0
    output = {
        "percent": percent,
        "total": total,
        "total_in_tumor": total_in_tumor,
        "total_pos_in_tumor": total_pos_in_tumor,
        "total_pos_out_tumor": total_pos_out_tumor,
        "not_in_tiles": not_in_tiles,
    }
    return output


def check_tile_near_feature(feat, tile_xy, tile_size):
    """feat = geojson qupath annotation feature
    tile_xy = list of [x,y] coordinates
    returns: whether points are in ANY polygon of this feature
    """
    coords = feat["geometry"]["coordinates"]
    nearby = False
    thresh = math.sqrt((tile_size**2) * 2) * 2
    if feat["geometry"]["type"] != "MultiPolygon":
        coords = [coords]
    for i, multi_polygon in enumerate(coords):
        for ii, polygon in enumerate(multi_polygon):
            if len(polygon) > 2:
                dat = np.array(polygon)
                min_x = np.min(np.sqrt((tile_xy[0] - dat[0, :]) ** 2))
                min_y = np.min(np.sqrt((tile_xy[1] - dat[1, :]) ** 2))
                if (min_x < thresh) | (min_y < thresh):
                    nearby = True
                    break
    return nearby, i, ii


def check_overlap_feat(feat, points):
    in_poly = check_points_in_feature(feat, points)
    return np.sum(in_poly) / len(in_poly) * 100


def check_tile_overlap_feat(feat, tile_xy, tile_size):
    points = []
    per_overlap = 0
    nearby, n_mp, n_p = check_tile_near_feature(feat, tile_xy, tile_size)
    if nearby:
        for y in range(tile_xy[1], tile_xy[1] + tile_size):
            for x in range(tile_xy[0], tile_xy[0] + tile_size):
                points.append([x, y])
        per_overlap = check_overlap_feat(feat, points)
    return per_overlap


def check_points_in_feature(feat, points):
    """feat = geojson qupath annotation feature
    points = list of [x,y] coordinates
    returns: whether points are in ANY polygon of this feature
    """
    coords = feat["geometry"]["coordinates"]

    if feat["geometry"]["type"] != "MultiPolygon":
        coords = [coords]
    all_out = []
    for ii, multi_polygon in enumerate(coords):
        for i, polygon in enumerate(multi_polygon):
            # out = np.zeros((len(points),1))
            path = mpl.path.Path(polygon)
            inside2 = path.contains_points(points)
            all_out.append(inside2)
    all_out = np.array(all_out)
    return np.any(all_out, axis=0)


def parse_tile_fn(fn):
    # tile_fn = '%s_n%d_x%d_y%d_px%d.jpg' % (slide_num,i,x,y,tile_size)
    p = fn.split(".")[0].split("_")
    for e in p:
        if e[0] == "x":
            x = int(e[1:])
        elif e[0] == "y":
            y = int(e[1:])
        elif e[0] == "p":
            size = int(e[2:])
        elif e[0] == "n":
            n = int(e[1:])
    return n, x, y, size


def plot_tile_on_annotation(feat, tile_xy, tile_size, ax):
    ax = plot_geojson_feature(feat, ax)
    polygon = feat["geometry"]["coordinates"]
    points = []
    for y in range(tile_xy[0], tile_xy[0] + tile_size):
        for x in range(tile_xy[1], tile_xy[1] + tile_size):
            points.append([x, y])
    in_poly = check_points_in_feature(feat, points)
    idxs = [in_poly == True, in_poly == False]
    pa = np.array(points)
    for i, idx in enumerate(idxs):
        xy = pa[idx, :]
        if i == 0:
            ax.plot(xy[:, 0], xy[:, 1], "r.")
        else:
            ax.plot(xy[:, 0], xy[:, 1], "y.")
    ax.set_title(
        "Test area %2.1f%% in annotation." % (np.sum(in_poly) / len(in_poly) * 100)
    )


def add_coords_to_tile_df(tile_df, swap_xy=True):
    """Take in tile inference dataframe from inference pipeline (tile_df)
    e.g.  pd.read_csv(infer_path.joinpath('slide_7_all_valid_pred.csv'))
    # add tile coordiantes to this.
    Note: Depending on version, these coordinates are swapped (x,y) relative to other qupath exported coords (e.g. annotations or tma array)
    tile_pred = add_coords_to_tile_df(tile_pred)
    tile_pred.head()
    Unnamed: 0	cur_path	slide	slide_class	p_pos	pred_cls	y	x	width
    187	187	/ix/rbao/Projects/WSI-REG-000-BIsett-RBao/resu...	LARYNX 6-2_H&E	NaN	0.000007	0	14560	14560	224
    250	250	/ix/rbao/Projects/WSI-REG-000-BIsett-RBao/resu...	LARYNX 6-2_H&E	NaN	0.041106	0	13216	13328	224
    """
    if swap_xy:  # True for now
        tile_df.loc[:, "y"] = (
            tile_df.cur_path.str.split("_").str[4].str[1:].astype(int)
        )  # Reversed x & y
        tile_df.loc[:, "x"] = tile_df.cur_path.str.split("_").str[5].str[1:].astype(int)
    else:
        tile_df.loc[:, "x"] = (
            tile_df.cur_path.str.split("_").str[4].str[1:].astype(int)
        )  # Reversed x & y
        tile_df.loc[:, "y"] = tile_df.cur_path.str.split("_").str[5].str[1:].astype(int)

    tile_df.loc[:, "width"] = (
        tile_df.cur_path.str.split("_").str[6].str[2:].str.split(".").str[0].astype(int)
    )

    return tile_df


def unify_core_numbers(df, use_col="Core name", out_col="Core #"):
    """Convert Letter-Number or Number-Letter convetion to LetterNumber
    E.g. A-1 becomes A1
         2-B becomes B2
    use_col is column to unify
    out_col is new column to put in df
    """
    temp = df[use_col].str.replace("-", "")
    new = []
    for core in temp:
        if core[0].isdigit():
            core = "%s%s" % (core[1], core[0])
        new.append(core)
    df[out_col] = new
    return df


def tiles_in_core_df_subset(tma_dat, core, tile_df):
    """Return specific subset of tiles that overlap with a given core and normalize their coordinates to the core
    Given tile dataframe as described in add_coords_to_tile_df() AND exported qupath TMA dearrayed core locations
    (QuPath -> manually or using exportTMAData(project + img_name + '_core_data.qptma', 10) ->
    i.e. tma_dat = pd.read_csv('export.qptma', header=4, delimiter='\t')
    return subset of tile_df with tiles that fall inside the given core - 'core'
    and also add centered, core-normalized tile coordinates under norm_x and norm_y
    return subset of tile_df
    example:
    subset = find_tiles_in_core(tma_dat,'%s-%s' %(core[0],core[1]),tile_pred)
    subset.head()
    Unnamed: 0	cur_path	slide	slide_class	p_pos	pred_cls	y	x	width	norm_x	norm_y
    187	187	/ix/rbao/Projects/WSI-REG-000-BIsett-RBao/resu...	LARYNX 6-2_H&E	NaN	0.000007	0	14560	14560	224	3315.0	1929.0
    250	250	/ix/rbao/Projects/WSI-REG-000-BIsett-RBao/resu...	LARYNX 6-2_H&E	NaN	0.041106	0	13216	13328	224	2083.0	585.0
    298	298	/ix/rbao/Projects/WSI-REG-000-BIsett-RBao/resu...	LARYNX 6-2_H&E	NaN	0.999873	1	17024	14896	224	3651.0	4393.0
    """
    subset = pd.DataFrame([])
    idx = tma_dat.loc[:, "Core #"] == core
    if any(idx):
        x, y, width = list(tma_dat.loc[idx, ["X", "Y", "Width"]].values[0])
        # print(x,y)
        x = int(x)
        y = int(y)
        width = int(width)
        # print([x,y,width])
        tx = tile_df.x.values + (tile_df.width.values / 2)
        ty = tile_df.y.values + (tile_df.width.values / 2)
        idx1 = (tx > x) & (tx < (x + width))
        idx2 = (ty > y) & (ty < (y + width))
        # print(sum(idx1 & idx2))
        if any(idx1 & idx2):
            subset = tile_df.loc[idx1 & idx2, :].copy()
            subset.loc[:, "core"] = core
            subset.loc[:, "norm_x"] = (subset.x + subset.width / 2) - x
            subset.loc[:, "norm_y"] = (subset.y + subset.width / 2) - y
            subset = subset.reset_index(drop=True)

    return subset
