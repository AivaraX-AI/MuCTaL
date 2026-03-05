import sys
import re
import pandas as pd
import numpy as np
from pathlib import Path

# Identify data to use for inference (can be same path
# as trained_model_path for cross-validation on held-out training data or
# a path to new data organized in the same way)
infer_data_path = Path(sys.argv[1])  # /tiles or /summary level of data path
file_type = "*.jpg"

use_model = sys.argv[2]  # resnet18_10fold_10rep_500bal_5ft -> 10-fold cv
# resnet18_8train_100rep_500bal -> train with 8 slides
# see below for more details

model_folder = "%s_model" % infer_data_path.parts[-1]  #'tiles' or 'summary'
infer_path = infer_data_path.parent.joinpath(model_folder)
pred_csv_path = infer_path.joinpath("%s/infer_csv" % use_model)  # Assume it exists

# Calulate using winner-take-all?
use_wta = False

# Create some regular expression queries (turn this into a function):
re_class = r"class_(.+)_x\d+_y\d+.jpg"
re_slide = r"(.+)_class_\S+_x\d+_y\d+.jpg$"
re_slide_class = r"(.+)_class_(.+)_x\d+_y\d+.jpg$"
slides = []
slides_c = []
all_c = []
all_f = []

# Search preprocessed data folder for tiles and extract
# slides and pos/neg classes from file names

for f in infer_data_path.glob(pattern=file_type):
    all_f.append(f.name)
    slide, c = re.findall(re_slide_class, f.name)[0]
    slides_c.append("%s_%s" % (slide, c))
    slides.append(slide)
    all_c.append(c)
print(np.unique(slides_c), len(np.unique(slides_c)))
posf = np.array([(f, c, s) for f, c, s in zip(all_f, all_c, slides) if c == "pos"])
negf = np.array([(f, c, s) for f, c, s in zip(all_f, all_c, slides) if c == "neg"])

# initialize output dataframe
u_slides_c = np.unique(slides_c)
temp = [item.split("_") for item in u_slides_c]
fold_df = pd.DataFrame(temp, columns=["slide", "class"])

all_inf_csv = [x.parts[-1] for x in pred_csv_path.glob("*.csv")]
fold_reps = []
for fn in all_inf_csv:
    val = fn.split("_")[1]
    if "summary.csv" not in val:
        fold_reps.append(int(val))

# For each trained model, load a corresponding csv file (fold, rep, or new data source):
for fold_rep in fold_reps:
    fn = pred_csv_path.joinpath("fold_%d_all_valid_pred.csv" % fold_rep)
    print("Loading %s" % fn)
    test_df = pd.read_csv(fn)
    test_slides = np.unique(test_df.loc[:, "slide"])

    p = np.array(
        test_df.loc[:, "p_pos"]
    )  # Probability ['neg','pos'] (Can check with dls.vocab )
    c = np.array(test_df.loc[:, "pred_cls"])  # Predictions decoded

    fold_df["p1_fold_%d" % fold_rep] = np.zeros((fold_df.shape[0], 1)) * np.nan
    if use_wta:
        fold_df["wta_fold_%d" % fold_rep] = np.zeros((fold_df.shape[0], 1)) * np.nan
    for slide in test_slides:
        src_idx = np.array(test_df.loc[:, "slide"]) == slide
        p1 = np.mean(p[src_idx])
        dest_idx = np.array(fold_df.loc[:, "slide"]) == str(slide)
        fold_df.loc[dest_idx, "p1_fold_%d" % fold_rep] = p1
        if use_wta:
            if np.sum(c[src_idx] == 0) > np.sum(c[src_idx] == 1):
                wta = 0
            else:
                wta = 1
            fold_df.loc[dest_idx, "wta_fold_%d" % fold_rep] = wta
p1 = []
if use_wta:
    wta = []

for fold_rep in fold_reps:
    p1.append(np.array(fold_df.loc[:, "p1_fold_%d" % fold_rep]))
    if use_wta:
        wta.append(np.array(fold_df.loc[:, "wta_fold_%d" % fold_rep]))
p1 = np.array(p1).transpose()
m_p1 = np.nanmean(p1, axis=1)  # Average validation performance across model folds/reps
fold_df["mean_ppos"] = m_p1
fold_df["pred_is_pos_0p5"] = m_p1 > 0.5

fn = pred_csv_path.joinpath("fold_summary.csv")
print("Saving inference summary to %s" % fn)
fold_df.to_csv(fn)
print("Finished")
