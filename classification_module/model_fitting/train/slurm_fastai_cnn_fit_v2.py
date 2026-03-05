from fastai.vision.all import *
from sklearn.utils import resample
import sklearn.metrics as skm
from pathlib import Path
import numpy as np
from numpy import random
import sys
import os
import time


# Create some regular expression queries:
proj = Path(sys.argv[1])
ver = sys.argv[2]
path = proj.joinpath('results/v%s' % ver)
run_name = sys.argv[3] #i.e. densenet169_10fold_1rep_250bal_224px
fold = sys.argv[4]
arch = run_name.split('_')[0] #I.E. "resnet18" in "resnet18_2train_100rep_500bal"
rep=int(run_name.split('_')[2].split('rep')[0])
use_sing = False #Assume inside singularity image?

print('Training %s %s' % (path,run_name))
# Organization:
s=time.time()
model_path = 'models'
fold_model_path=path.joinpath(model_path).joinpath(run_name).joinpath('fold_models')
fold_model_path.mkdir(parents=True, exist_ok=True)
csv_path=path.joinpath(model_path).joinpath(run_name).joinpath('csv')
if csv_path.exists() == False:
    print('Folder of CV folds as .csv files not found!')
    
# infer_path=path.joinpath(model_path).joinpath(run_name).joinpath('infer_csv')

#Train model on as many folds as read in from SLURM batch call:
print('Fold :', fold )
df=pd.read_csv(csv_path.joinpath('train_valid_fold_%s.csv' % fold))

#Generate correct root path for v8 project run on HTC:
new_pns=[]
for img_fn in df.fn:
    temp = img_fn.split('.')[0]
    px = temp.split('_')[-1].split('px')[1] + 'px'
    slide = temp.split('_')[0]
    new_pns.append(str(path.joinpath('tiles').joinpath(px).joinpath(slide).joinpath(img_fn)))
df.loc[:,'cur_path'] = new_pns
    
batch_size = 320 #Need > 30GB GPU to run densenet169 w/ this batch size
tissue =DataBlock(blocks=(ImageBlock, CategoryBlock),
                  get_x=ColReader('cur_path'),
                  splitter=ColSplitter('is_valid') ,
                  get_y=  ColReader('class'),
                  item_tfms=Resize(460), #Presize
                  batch_tfms=aug_transforms(size=224,
                                            max_rotate=45, # size=224,
                                            min_scale=1,
                                            max_zoom=0,
                                            flip_vert=True,
                                           )
                             ) 
dls = tissue.dataloaders(df, bs = batch_size)

print('Begin training:')
learn = cnn_learner(dls, eval(arch),
                metrics=[error_rate, accuracy],
                ).to_fp16()

#Updated for version 2:
learn.fit_one_cycle(40)
learn.fit_one_cycle(5,4e-3)
learn.unfreeze()
learn.fit_one_cycle(10,lr_max=1e-3)
fn = fold_model_path.joinpath('%s_%d_%s.pkl' % (arch,rep,fold))
print(fn)
learn.export(fname=fn)


try:
    interp = ClassificationInterpretation.from_learner(learn)
    upp, low = interp.confusion_matrix()
    tn, fp = upp[0], upp[1]
    fn, tp = low[0], low[1]
    print(tn, fp, fn, tp)
    sensitivity = tp/(tp + fn) #True pos / all positive
    specificity = tn/(fp + tn)
    print('specificity',specificity) #True neg / all negative
    print('sensitivity',sensitivity)
    print('false positive', fp/ (fp + tn))
    print('true positive', tp/(tp + fn) )
except:
    print('Unable to calculate performance.')
#        dl = learn.dls.test_dl(test_df.loc[:, 'full_path'])
#     print('Beginning inference:')
#     pred = learn.get_preds(dl=dl, with_decoded=True)
    
ss=time.time()
print('Wall time: %ds' % (ss-s))
print('Finished')
