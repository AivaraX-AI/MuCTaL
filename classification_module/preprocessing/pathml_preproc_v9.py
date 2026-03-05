import os
import sys
import cv2  # For blur detection
import json
import math
import time
import datetime
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from pathml.core import HESlide
from pathml.preprocessing import (
    TissueDetectionHE,
    LabelWhiteSpaceHE,
    LabelArtifactTileHE,
    StainNormalizationHE,
)

debug = False  # Only process 1000 tiles
use_dask = False  # Currently not sure what is going on with scheduler
version = 9  #
# Version 4 included LabelArtifactTileHE
# Version 5 includes blurry image exclusion via laplacian variance coefficient (cv2.Laplacian), bloodclot removal based on RGB values
# Version 6 adds Macenko tile-level color correction -- no bloodclot remove, "summary" slides thresholded for % tissue to use for large-tile model
# Version 8 -> MANY changes. remove large tile segmentation (does not help),
# ->take tile size in as arg. segmentation.
# -> Take in a list of files specificed from a spreadsheet. No clot detection/removal,
# -> tile-level color correction -> keep for now, add kernel blur to tissue detect
# -> increase tile jpg quality to -> 95 <- [only do this when using very small tiles?] currently for all tiles
# Version 9 --> remove unnecessary variable inputs: src, etc
# -> apply normalize to tile after tissue detect (improved stain norm?)
# -> flip (x,y) coordinates because they were wrong previously
# -> major update: if anno_path column in the sample.tsv file, and the annotation .geojson exists for that file, attempt to assign tiles to the annotations found in the geojson (>10% overlap)

# Begin the job
dest = Path(sys.argv[1])  # e.g. /mnt/results/
file_list_pnfn = sys.argv[
    2
]  # Sampleinfo spreadsheet destination e.g. 'new_data_20220829_diagnosis_data.tsv'
job_id = int(sys.argv[3])  # pick file to use
tile_size = int(sys.argv[4])
helper_path = sys.argv[
    5
]  #'/ix/rbao/Projects/panCancer_HE/scripts/pancancer_he_classifier'
sys.path.append(helper_path)
from helpers import anno as annoHelper

tile_stride = tile_size  # tile_size//2
tissue_thresh = 70  # % of tile that contains tissue
# Stain norm will work better with less empty tile
gj_anno_overlap = 10  # If assigning tiles to geojson this percent of tile must be inside annotation to be included
blur_thresh = 40  # threshold for blurriness calculated by strength of edges in image (higher = sharper image)
use_stain_norm = True

start = time.time()  # Important to catch pathml load time as it can be quite long
print("\nInside preproc v%d" % version)

if debug == True:
    print("Running in debug mode!")
    print("Inputs received:")
    print("\n\t".join(sys.argv))
else:
    print("Running in processing mode (not debug).")

info = pd.read_csv(file_list_pnfn, sep="\t")
tile_dest = (
    dest.joinpath("v%d" % version).joinpath("tiles").joinpath("%dpx" % tile_size)
)
time_start_str = time.strftime("%Y-%m-%d %H:%M:%S%p", time.localtime(start))

print(
    "Uses: blank tile detection-> artifact detection -> tissue > %d%%, for final tile inclusion"
    % tissue_thresh
)
print("Job log beginning at %s" % (time_start_str))
print("Sampleinfo table: %s\n" % str(file_list_pnfn))

use_slide = str(info.slide.values[job_id])
print("Use slide", use_slide, type(use_slide))
tile_dest = (
    dest.joinpath("v%d" % version)
    .joinpath("tiles")
    .joinpath("%dpx" % tile_size)
    .joinpath(use_slide.split(".")[0])
)
fn = Path(info.slide_path.values[job_id])
file_avail = fn.exists()
print("File available", file_avail)
use_anno = False
tile_df = pd.DataFrame([])
if "anno_path" in info.columns:
    print("Annotation file column detected.")
    gj_fn = Path(info.anno_path.values[job_id])
    if gj_fn.exists():
        use_anno = True
        print(
            "annotation file found: will check overlaps", info.annotation.values[job_id]
        )
        print("Loading geojson...")
        with open(gj_fn) as f:
            allobjects = json.load(f)  #
    else:
        print("%s not found" % str(gf_fn))


# Color layer rearrange:
bgr = (2, 1, 0)
if file_avail:
    slide_num = info.slide.values[job_id].split(".")[0]
    print("This job will process %s" % (info.slide[job_id]))

    tile_dest.mkdir(parents=True, exist_ok=True)

    # Create whole slide image object:
    wsi = HESlide(str(fn))

    print("\t\tBeginning to preprocess file (not distributed).")
    tissue_detect = TissueDetectionHE(
        mask_name="tissue", outer_contours_only=True, blur_ksize=21, threshold=20
    )  # Thresh possibly off for smaller tile size (<=224 )?

    blank_detect = LabelWhiteSpaceHE(
        label_name="blank",
        proportion_threshold=0.9,  # Thresh too low?
    )

    art_detect = LabelArtifactTileHE(label_name="artifact")
    normalizer = StainNormalizationHE(
        target="normalize", stain_estimation_method="macenko"
    )

    # TILE WSI AND FILTER TILES BASED ON VARIOUS CRITERIA:
    tiles = wsi.generate_tiles(shape=tile_size, stride=tile_stride, pad=False, level=0)

    tot = 127 * tile_size**2
    tot_tiles_est = (
        (wsi.shape[0] // tile_size)
        * (wsi.shape[1] // tile_size)
        * (2 * (tile_size // tile_stride))
    )

    # Does not include all the filtering to be done
    print(
        "\t\tSave tiles at %dx%d pixel size (tissue only > %d%%), stride = %d"
        % (tile_size, tile_size, tissue_thresh, tile_stride)
    )
    print(
        "\t\tTotal tile estimate (without filtering blank tiles etc): %d"
        % tot_tiles_est
    )

    # Run the actual pipeline (the slow part):

    ii = 0
    for i, tile in enumerate(tiles):
        blank_detect.apply(tile)
        if tile.labels["blank"] == False:
            art_detect.apply(tile)

            # In addition -- detect pen / glass edge artifacts??
            if tile.labels["artifact"] == False:
                im = np.array(tile.image)  # r g b page order
                blur_detect = cv2.Laplacian(im[:, :, bgr], cv2.CV_64F).var()

                if blur_detect > blur_thresh:
                    tissue_detect.apply(tile)
                    tissue_mask = tile.masks["tissue"]
                    per = 100 * np.sum(tissue_mask.flatten()) / tot

                    if per > tissue_thresh:
                        if use_stain_norm:
                            normalizer.apply(tile)
                            im = np.array(tile.image)  # r g b page order
                        y, x = tile.coords  # Flipped to match qupath x,y in v9
                        tile_fn = "%s_n%d_x%d_y%d_px%d.jpg" % (
                            slide_num,
                            i,
                            x,
                            y,
                            tile_size,
                        )
                        tile_df.loc[ii, "tile"] = tile_fn
                        tile_df.loc[ii, "x"] = x
                        tile_df.loc[ii, "y"] = y
                        print(
                            "\t\t\t--%d) Saving tile %s (%d/%d)"
                            % (ii, tile_dest.joinpath(tile_fn), i, tot_tiles_est)
                        )

                        img_fn = tile_dest.joinpath(tile_fn)
                        data = Image.fromarray(im.astype(np.uint8))
                        data.save(img_fn, quality=95)

                        # Potentially: if geojson associated with wsi, determine if tile overlaps with annotations
                        if use_anno:
                            max_overlap = 0
                            # or enumerate(allobjects['features']) ? how to standardize?
                            if isinstance(
                                allobjects, list
                            ):  # list of objects, possibly multipolygons
                                enum_obj = allobjects
                            elif (
                                "features" in allobjects.keys()
                            ):  # flat list of annotations
                                enum_obj = allobjects["features"]
                            for iii, feat in enumerate(
                                enum_obj
                            ):  # Be aware that this structure can change considerably
                                if "classification" in feat["properties"].keys():
                                    anno_type = feat["properties"]["classification"][
                                        "name"
                                    ]
                                else:
                                    anno_type = "anno"
                                print("Finding tiles in %s" % anno_type)
                                per_overlap = annoHelper.check_tile_overlap_feat(
                                    feat, [x, y], tile_size
                                )
                                if per_overlap > max_overlap:
                                    max_overlap = per_overlap
                                    use_anno = anno_type
                                    use_feat = iii
                            if max_overlap > gj_anno_overlap:
                                # To deal with cases with more than one annotation in a tile:
                                tile_df.loc[ii, "anno"] = use_anno
                                tile_df.loc[ii, "anno_num"] = use_feat
                                tile_df.loc[ii, "overlap"] = max_overlap
                            else:
                                tile_df.loc[ii, "anno"] = ""
                                tile_df.loc[ii, "anno_num"] = np.nan
                                tile_df.loc[ii, "overlap"] = max_overlap

                        cur_time = time.time()
                        dur = str(datetime.timedelta(seconds=(cur_time - start)))
                        print(
                            "\t\t\t\t %3.1f%% complete. %s H:M:S"
                            % (i / tot_tiles_est * 100, dur)
                        )
                        ii += 1  # Counter of included tiles (excludes blanks/artifacts/blur etc)
                        if (debug == True) and (ii > 100):
                            print("Debug max of 100 tiles reached, exiting.")
                            break
    dfn = "%s_%d_tiles_df.tsv" % (slide_num, tile_df.shape[0])
    tile_df_pnfn = dest.joinpath("v%d" % version).joinpath("tiles").joinpath(dfn)
    print("Saving", tile_df_pnfn)
    tile_df.to_csv(tile_df_pnfn, sep="\t")
    print("Processing complete.")
else:
    print("No %s files found in %s" % (file_type, str(src)))

stop = time.time()
dur = str(datetime.timedelta(seconds=(stop - start)))
print("Wall time = %s H:M:S" % dur)
