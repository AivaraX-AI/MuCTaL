from fastai.vision.all import *
from pathlib import Path
import numpy as np
import shutil
import glob
import os
import pandas as pd
from torch import cuda
import shutil
import time
from tqdm import tqdm

state = 36
scratch = os.getenv("SLURM_SCRATCH")
copy_f = False
# Copy to scratch:
print("Copy tiles to %s" % scratch)
df = pd.read_csv(
    "/ix/rbao/Projects/panCancer_HE/data/balanced_10000_sk_lu_lv_cr_df_v1_79984tiles.tsv",
    sep="\t",
)
df.loc[:, "tissue_anno"] = df.tissue + df.anno
df.loc[:, "scratch_fn"] = scratch + "/" + df.fn.str.split("/").str[-1]

if copy_f:
    for i, fn in enumerate(tqdm(df.fn.values)):
        scratch_fn = df.loc[i, "scratch_fn"]
        try:
            shutil.copyfile(fn, scratch_fn)
        except:
            print(scratch_fn, "Copy Failed")
print(df.scratch_fn.isna().sum(), "missing")
print(df.groupby(["tissue", "anno"])["fn"].count())

# Make data loader:
state = 36
splitter = TrainTestSplitter(
    test_size=0.1,
    random_state=state,
    stratify=df.tissue_anno.values,
    train_size=None,
    shuffle=True,
)
batch_size = 375  # Requires A100
tissue = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_x=ColReader("scratch_fn"),
    splitter=splitter,
    get_y=ColReader("anno"),
    item_tfms=Resize(460),  # Presize
    batch_tfms=[
        *aug_transforms(
            size=224,
            max_rotate=45,  # size=224,
            min_scale=1,
            max_zoom=0,
            flip_vert=True,
        ),
        Normalize.from_stats(*imagenet_stats),
    ],
)
dls = tissue.dataloaders(df, bs=batch_size)


learn = cnn_learner(
    dls,
    densenet169,
    metrics=[accuracy],  # AccumMetric() Recall() ?
).to_fp16()

# # Figure out what is reproducibly good:
learn.fit_one_cycle(10)  # 10 seems better than 20
learn.unfreeze()
learn.fit_one_cycle(20, lr_max=3e-3)  # worked well
learn.fine_tune(5, base_lr=1e-5)  # 10 seemed too long
learn.fine_tune(5, base_lr=5e-6)


fn = Path(
    "/ix/rbao/Projects/panCancer_HE/results/models/densenet169_bs%d_n%d/"
    % (batch_size, df.shape[0])
)
fn.mkdir(exist_ok=True)
learn.export(fn.joinpath("full_model.pkl"))
