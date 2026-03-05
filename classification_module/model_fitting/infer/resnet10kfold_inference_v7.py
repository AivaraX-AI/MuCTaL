''' fastai model inference script v7

'''

import os
import sys
import time
import numpy as np
from pathlib import Path
from fastai.vision.all import *

# Variables passed in from infer_path_v{n}.job, n >= 7
trained_model_path = Path(sys.argv[1])  # /tiles or /summary level of data path

# Identify data to use for inference (can be same path
# as trained_model_path for cross-validation on held-out training data or
# a path to new data organized in the same way)
infer_data_path = Path(sys.argv[2])  # /tiles or /summary level of data path
# e.g. .../data/proc/v6/tiles (or comparable)
file_type = '*.jpg'  # image type of tiles or summary preprocessed data

use_model = sys.argv[3]  # resnet18_10fold_10rep_500bal_5ft -> 10-fold cv
# resnet18_8train_100rep_500bal -> train with 8 slides
# see below for more details

# Split apart model architecture and parameters from use_model
(arch, foldstr, repstr, balstr, ftstr) = use_model.split('_')
kfold = int(foldstr.split('fold')[0])
nrep = int(repstr.split('rep')[0])
nbal = int(balstr.split('bal')[0])
nft = int(ftstr.split('ft')[0])


print('Analyzing %s %s images' % (infer_data_path, file_type))

# Set fold/rep csv files to use for inference e.g. (0, 100) not inclusive for 10 fold, 10 rep
if len(sys.argv) > 4:
    # Infer data with a model indicated by jobid passed to sys.argv[4]
    start = int(sys.argv[4])
    stop = start + 1  # in batch evaluate a single fold/rep model
else:
    # Without this, assume there are 10 models / .csv files for 10-fold cv
    start = 0
    stop = 10

# Create some regular expression queries (turn this into a function):
re_class = r"class_(.+)_x\d+_y\d+.jpg"
re_slide = r'(.+)_class_\S+_x\d+_y\d+.jpg$'
re_slide_class = r'(.+)_class_(.+)_x\d+_y\d+.jpg$'
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
posf = np.array([(f, c, s) for f, c, s in zip(all_f, all_c, slides) if c == 'pos'])
negf = np.array([(f, c, s) for f, c, s in zip(all_f, all_c, slides) if c == 'neg'])

# initialize output dataframe
u_slides_c = np.unique(slides_c)
temp = [item.split('_') for item in u_slides_c]
fold_df = pd.DataFrame(temp, columns=['slide', 'class'])

# Set timer
s = time.time()

# Point to correct model and data paths:
model_folder = '%s_model' % trained_model_path.parts[-1] #'tiles' or 'summary'
train_path = trained_model_path.parent.joinpath(model_folder)
infer_path = infer_data_path.parent.joinpath(model_folder)

fitted_model_path = train_path.joinpath('%s/fold_models' % use_model) #Path must already exist
csv_path = infer_path.joinpath('%s/csv' % use_model) #created by make_crossval_csv_v7.py
pred_csv_path = infer_path.joinpath('%s/infer_csv' % use_model) #Path can be new path if new data

if not pred_csv_path.exists():
    os.makedirs(pred_csv_path)

# For each trained model, load a corresponding csv file (fold, rep, or new data source):
for fold_rep in range(start, stop):
    print('Perform inference on fold/rep :', fold_rep)

    # Load trained model:
    model_fn = fitted_model_path.joinpath('%s_1_%d_%d.pkl' % (arch, nft, fold_rep))
    learn = load_learner(model_fn)

    # Read in data to use for inference from matching fold_rep .csv file
    # (see:  make_crossval_csv_v{n}.py )
    df = pd.read_csv(csv_path.joinpath('train_valid_fold_%d.csv' % fold_rep))
    test_df = df.loc[df.loc[:, 'is_valid'] == 1, :].reset_index(drop=True)
    dl = learn.dls.test_dl(test_df.loc[:, 'full_path'])
    print('Beginning inference:')
    pred = learn.get_preds(dl=dl, with_decoded=True)

    test_slides = np.unique(test_df.loc[:, 'slide'])
    p = np.array(pred[0])  # Probability ['neg','pos'] (Can check with dls.vocab )
    c = np.array(pred[2])  # Predictions decoded

    # Save all predictions:
    test_df['p_pos'] = p[:, 1]
    test_df['pred_cls'] = c
    fn = pred_csv_path.joinpath('fold_%d_all_valid_pred.csv' % fold_rep)

    print('Saving all validation inferences to %s' % fn)
    test_df.to_csv(fn)

ss = time.time()
print('Wall time: %ds' % (ss - s))
print('Finished')
