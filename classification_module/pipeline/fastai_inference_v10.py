''' fastai model inference script v10
    V8 -- assumes tiles are in a different folder structure-> results/v8/tiles/pixelsize/SLIDENUM/files.jpg
    V10 -- designed to work in pipeline, only expectation is an input list of tiles to infer in specific format (one output of preproc)
'''
import os
import sys
import time
import datetime
import numpy as np
from tqdm import tqdm
from pathlib import Path
from fastai.vision.all import *

tile_df_pn = Path(sys.argv[1]) #
model_fn = Path(sys.argv[2]) #Trained model path (can be same as data path)
tile_path = sys.argv[3]
output = Path(sys.argv[4])  #infer_data_path v{n}/tiles/'224px' or blah/'500px' etc. but will assume [vn]/parse/[type] for version and type

s = time.time()
df = pd.read_csv(tile_df_pn, sep='\t')   
df.loc[:,'cur_path'] = tile_path + df.tile
pred_csv_path = output.joinpath('infer_csv') #Path can be new path if new data
print('Output path:', pred_csv_path)
pred_csv_path.mkdir(parents=True, exist_ok=True)

# Load trained model:
learn = load_learner(model_fn)  #Run with GPU to get time improvement? : learn = load_learner(model_fn, cpu = False)
dl = learn.dls.test_dl(df.loc[:,'cur_path'], num_workers=4)
print('Beginning inference:')
pred = learn.get_preds(dl=dl, with_decoded=True) 

# Add tile prediction as column to tile df:
p = np.array(pred[0])  # Probability ['neg','pos'] (Can check with dls.vocab )
c = np.array(pred[2])  # Predictions decoded
df.loc[:,'p_pos'] = p[:, 0] # ['Tumor', 'notTumor'] (via dls.vocab)
df.loc[:,'pred_cls'] = c
fn = output.joinpath('infer_%s' % (tile_df_pn.parts[-1]))

# Save
print('Saving tile inference to %s' % fn)
df.to_csv(fn, sep ='\t', index=False)
ss = time.time()
#dur = str(datetime.timedelta(seconds=(ss-s)))
print('Inference Wall time = %03.1f minutes' % ((ss-s)/60))
print('Inference Finished')
