import os
import sys
import time
import numpy as np
from pathlib import Path
from fastai.vision.all import *

#

split_data_path = Path(sys.argv[1])
print("DATA_PATH:", split_data_path)
data_root = split_data_path.parent.parent
print("DATA_ROOT:", data_root)
trained_model_path = Path(sys.argv[2])  # Trained model path (can be same as data path)
print("TRAINED_MODEL_PATH:", trained_model_path)

use_model = trained_model_path.parent.parts[
    -1
]  # aka run_name i.e. "resnet18_jackknife", "resnet18_10fold_10rep_500bal_5ft", resnet18_2train_100rep_25bal
print("INFER_MODEL:", use_model)
slide_df_fn = sys.argv[3]
# Set fold/rep csv files to use for inference e.g. (0, 100) not inclusive for 10 fold, 10 rep
if len(sys.argv) > 4:
    # Infer data with a model indicated by jobid passed to sys.argv[4]
    start = int(sys.argv[4])
    stop = start + 1  # in batch evaluate a single fold/rep model
else:
    # Without this, assume there are 10 models / .csv files for 10-fold cv
    start = 0
    stop = 10

if len(sys.argv) > 5:
    # Infer data with a model indicated by jobid passed to sys.argv[4]
    method = sys.argv[5]
else:
    method = "FOLDS"  # default. other method = SLIDES

tile_version = split_data_path.parts[-3]
tile_type = split_data_path.parts[-1]
file_type = "*.jpg"

# 'use_model' string contains model architecture and parameters :
arch, foldstr, repstr, balstr, pxstr, ver = use_model.split("_")
kfold = int(foldstr.split("fold")[0])  # kfolds
nrep = int(repstr.split("rep")[0])  # n repeitions of k-fold
nbal = int(balstr.split("bal")[0])
npx = int(pxstr.split("px")[0])  # n pixels of image size
ver = int(ver.split("v")[1])  # Version of model run

# Load list of all tiles:
slide_df = pd.read_csv(slide_df_fn, sep="\t")

# Set timer
s = time.time()
model_path = "models"
# trained_model_path = path.joinpath('fold_models')
csv_path = data_root.joinpath(model_path).joinpath(use_model).joinpath("csv")
pred_csv_path = (
    data_root.joinpath(model_path).joinpath(use_model).joinpath("infer_csv")
)  # Path can be new path if new data
print("Output path:", pred_csv_path)
pred_csv_path.mkdir(parents=True, exist_ok=True)

# For each trained model, load a corresponding csv file (fold, rep, or new data source):
for fold_rep in range(start, stop):
    if method == "FOLDS":
        print("Perform inference on fold/rep :", fold_rep)

        # Load trained model:
        model_fn = trained_model_path.joinpath("%s_%d_%d.pkl" % (arch, nrep, fold_rep))
        learn = load_learner(model_fn)

        # Read in data to use for inference from matching fold_rep .csv file
        # (see:  make_crossval_csv_v{n}.py )
        df = pd.read_csv(csv_path.joinpath("train_valid_fold_%d.csv" % fold_rep))

        # Identify validation SLIDES:
        valid_slides = df.loc[df.is_valid.values == 1, "slide_num"].unique()
        save_type = "fold"
    elif "SLIDES" in method:
        # Perform inference on slides listed in slide_df using one full model.
        print("Perform inference on slide :", fold_rep)

        # Load trained model:
        model_fn = trained_model_path.joinpath("%s_%d_%d.pkl" % (arch, 1, 0))
        learn = load_learner(model_fn)

        # Identify validation SLIDES:
        valid_slides = [slide_df.loc[fold_rep, "slide_num"]]
        save_type = "slide"

    # Generate correct root path for v8 project run on HTC:
    print(valid_slides)
    valid_dict = {"cur_path": [], "slide": [], "slide_class": []}
    for slide in valid_slides:
        fns = [str(x) for x in split_data_path.joinpath(str(slide)).glob(file_type)]
        if "group" in slide_df.columns:
            slide_class = slide_df.group[slide_df.slide_num.values == slide].to_list()
        else:
            slide_class = ["NA"]
        valid_dict["cur_path"].extend(fns)
        valid_dict["slide"].extend([slide] * len(fns))
        valid_dict["slide_class"].extend(slide_class * len(fns))
    valid_df = pd.DataFrame(valid_dict)

    dl = learn.dls.test_dl(valid_df.loc[:, "cur_path"])
    print("Beginning inference:")
    pred = learn.get_preds(
        dl=dl, with_decoded=True
    )  # Run with GPU to get time improvement?

    test_slides = valid_slides
    p = np.array(pred[0])  # Probability ['neg','pos'] (Can check with dls.vocab )
    c = np.array(pred[2])  # Predictions decoded

    # Save all predictions:
    valid_df.loc[:, "p_pos"] = p[:, 1]
    valid_df.loc[:, "pred_cls"] = c
    fn = pred_csv_path.joinpath("%s_%d_all_valid_pred.csv" % (save_type, fold_rep))

    print("Saving all validation inferences to %s" % fn)
    valid_df.to_csv(fn)

ss = time.time()
print("Wall time: %ds" % (ss - s))
print("Finished")
