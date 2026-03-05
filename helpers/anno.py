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
    """_summary_

    Parameters
    ----------
    feat : _type_
        _description_
    ax : _type_
        _description_
    class_cm : dict, optional
        _description_, by default {"Malignant": "r", "Benign Bile Duct": "b"}

    Returns
    -------
    _type_
        _description_
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
    """_summary_

    Parameters
    ----------
    feat : _type_
        _description_
    subset : _type_
        _description_
    tma_dat : _type_
        _description_
    rotate_180 : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
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
    """_summary_

    Parameters
    ----------
    allobjects : _type_
        _description_
    subset : _type_
        _description_
    tma_dat : _type_
        _description_
    ax : _type_, optional
        _description_, by default None
    use_plot : bool, optional
        _description_, by default False
    class_cm : dict, optional
        _description_, by default {"Positive": "yellow", "Negative": "cyan"}
    thresh : float, optional
        _description_, by default 0.95

    Returns
    -------
    _type_
        _description_
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
    """_summary_

    Parameters
    ----------
    feat : _type_
        _description_
    tile_xy : _type_
        _description_
    tile_size : _type_
        _description_

    Returns
    -------
    _type_
        _description_
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
    """_summary_

    Parameters
    ----------
    feat : _type_
        _description_
    points : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    in_poly = check_points_in_feature(feat, points)
    return np.sum(in_poly) / len(in_poly) * 100


def check_tile_overlap_feat(feat, tile_xy, tile_size):
    """_summary_

    Parameters
    ----------
    feat : _type_
        _description_
    tile_xy : _type_
        _description_
    tile_size : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
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
    """_summary_

    Parameters
    ----------
    feat : _type_
        _description_
    points : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    coords = feat["geometry"]["coordinates"]

    if feat["geometry"]["type"] != "MultiPolygon":
        coords = [coords]
    all_out = []
    for ii, multi_polygon in enumerate(coords):
        for i, polygon in enumerate(multi_polygon):
            path = mpl.path.Path(polygon)
            inside2 = path.contains_points(points)
            all_out.append(inside2)
    all_out = np.array(all_out)
    return np.any(all_out, axis=0)


def parse_tile_fn(fn):
    """_summary_

    Parameters
    ----------
    fn : function
        _description_

    Returns
    -------
    _type_
        _description_
    """
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
    """_summary_

    Parameters
    ----------
    feat : _type_
        _description_
    tile_xy : _type_
        _description_
    tile_size : _type_
        _description_
    ax : _type_
        _description_
    """
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
    """_summary_

    Parameters
    ----------
    tile_df : _type_
        _description_
    swap_xy : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_
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
    """_summary_

    Parameters
    ----------
    df : _type_
        _description_
    use_col : str, optional
        _description_, by default "Core name"
    out_col : str, optional
        _description_, by default "Core #"

    Returns
    -------
    _type_
        _description_
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
    """_summary_

    Parameters
    ----------
    tma_dat : _type_
        _description_
    core : _type_
        _description_
    tile_df : _type_
        _description_

    Returns
    -------
    _type_
        _description_
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
