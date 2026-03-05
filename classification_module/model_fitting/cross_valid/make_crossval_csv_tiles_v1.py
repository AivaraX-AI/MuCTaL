from sklearn.model_selection import RepeatedKFold, LeaveOneGroupOut, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.utils import resample
from pathlib import Path
import numpy as np
import sys
import os
import re
import pandas as pd
ver = 10 


split_data_path = Path(sys.argv[1])  
tile_version = split_data_path.parts[-3]
tile_type = split_data_path.parts[-1]

use_model = sys.argv[2]  # i.e. "resnet18_jackknife", "resnet18_10fold_10rep_500bal_5ft", resnet18_2train_100rep_25bal  

# 'use_model' string contains model architecture and parameters :
(arch, foldstr, repstr, balstr, pxstr, ver) = use_model.split('_')
print(arch, foldstr, repstr, balstr, pxstr, ver)
kfold = int(foldstr.split('fold')[0])  # kfolds
nrep = int(repstr.split('rep')[0])  # replace with seed for random shuffling as opposed to rep ver 9
nbal = int(balstr.split('bal')[0])
# nft = int(ftstr.split('ft')[0]) #n finetuning epochs unused
npx = int(pxstr.split('px')[0]) #n pixels of image size
ver = int(ver.split('v')[1]) #Version of model run

# 'newdata' or 'crossvalid'
fold_method = sys.argv[3].split('_')[0]
bal_target = sys.argv[3].split('_')[1]

seed = 35

#Instead, load all_dat and/or target_df:
if bal_target == 'slide':
    #First balance per slides
    target_df_fn = sys.argv[4] # /mnt/sampleinfo/target_df_v8.0_44.tsv
    target_df = pd.read_csv(target_df_fn, sep = '\t')
else:
    # Create list of unique values in the appropriate field:
    target_df = pd.data
    
pos_class = target_df.group[target_df.loc[:,'anno']==True][0]
all_dat_fn = sys.argv[5] #/mnt/sampleinfo/tile_df_v8.0_255347.tsv
all_dat = pd.read_csv(all_dat_fn, sep='\t',dtype='str')

# Create export .csv folder
output_cv_csv_folder = 'csv' #Check where this should be for verison 8
model_folder = '%s_model' % split_data_path.parts[-1] #'224px' or '500px'
base_infer_path = split_data_path.parent.joinpath(model_folder)
csv_path = base_infer_path.parent.parent.joinpath('models/%s/%s' % (use_model, 
                                                             output_cv_csv_folder))
if not csv_path.exists():
    os.makedirs(csv_path)

if fold_method == 'crossvalid':  # Otherwise include all data for test (see below)
    # Create cross validation object
    ntarget = nbal  # Tiles per target if balance_tiles_per_target == True
    ntargets = [ntarget, ntarget]  # This can change depending on crossval_method see below

    if nbal == 0:
        # How to balance training example tiles during training and testing:
        balance_classes_total_tiles = [False, False]  # Train, Test - whether to balance to nbal
        balance_tiles_per_target = [False, False]  # Train, Test - whether to balance to nbal    
    else:
        # How to balance training example tiles during training and testing:
        balance_classes_total_tiles = [True, True]  # Train, Test - whether to balance to nbal
        balance_tiles_per_target = [True, True]  # Train, Test - whether to balance to nbal

    crossval_method = use_model.split('_')[1]
    print('Using %s cross-validation '
          'and fold # used as seed for fold-level resampling.\n'
          'Version %d.\nPositive class detected as %s.'
          % (crossval_method, ver,pos_class))

    if 'fold' in crossval_method: #Default for most models
        splitter = StratifiedGroupKFold(n_splits=kfold,
                                        shuffle = True,
                                        random_state=seed)
        
        print('With %d random_state and %d splits/folds' % (seed, kfold))

    elif 'train' in crossval_method:
        # Similar to k-fold but can set training size of n targets arbitrarily
        # Used for generating learning curve
        # e.g. resnet18_2train = train model using only 2 targets.

        balance_tiles_per_target = [True, True]  # Train, Test -- balance n tiles drawn from each target
        n_repeats = nrep  # In reality unique combinations limited by train_size & random_state
        k = int(crossval_method.split('train')[0])
        ntrain_tiles = nbal  # Use value found in use_model input string
        ntest_tiles = 500  # Keep this constant (for now)
        ntargets = [ntrain_tiles, ntest_tiles]
        splitter = GroupShuffleSplit(n_splits=1000,
                                     train_size=k,
                                     random_state=seed)
        print('With %d repeats and %d splits(folds)' % (n_repeats, k))

    elif crossval_method == 'jackknife':
        # Leave-one-out jack-knife
        splitter = LeaveOneGroupOut()



    ktrain = []
    ktest = []
    utrain = []

    for target_train_idx, target_test_idx in splitter.split(target_df.loc[:, bal_target],
                                                          target_df.loc[:, 'group'],
                                                          groups=target_df.loc[:, 'case']):
        if 'train' in crossval_method:
            # Make sure there are equal pos/neg classes in training this becomes very important with low k
            # I.e. there could be many splits with 0 examples of a class if this check is not performed!
            if (np.sum(target_df.loc[target_train_idx, 'anno'] == True) == k // 2) \
                    and (np.sum(target_df.loc[target_train_idx, 'anno'] == False) == k // 2):
                # If both classes have k/2 examples in the training set, keep them
                ktrain.append(target_train_idx)
                ktest.append(target_test_idx)
                utrain.append(str(np.sort(target_train_idx)))
            # Once the desired number of repeats is achieved, exit the loop and print n unique collected
            if len(ktrain) == n_repeats:
                print('Collected %d unique balanced training sets of size %d.'
                      % (len(np.unique(utrain)), k // 2))
                break
        else:
            ktrain.append(target_train_idx)
            ktest.append(target_test_idx)

    fold = 0  # Initialize
    for target_train_idx, target_test_idx in zip(ktrain, ktest):
        print('\nFold %d' % fold)
        sets = [target_train_idx, target_test_idx]
        set_lab = ['train', 'valid']
        df_list = []
        n_train_test = [np.sum(target_df.loc[target_train_idx, 'anno'] == True),  # n pos
                        np.sum(target_df.loc[target_train_idx, 'anno'] == False),  # n neg
                        np.sum(target_df.loc[target_test_idx, 'anno'] == True),  # n pos
                        np.sum(target_df.loc[target_test_idx, 'anno'] == False)]  # n neg
        print('Train targets n=%d pos, %d neg, Validation targets n=%d pos, %d neg' % tuple(n_train_test))
        valid_cases = target_df.loc[target_test_idx,'case'].unique()
        train_cases = target_df.loc[target_train_idx,'case'].unique()
        print('Validation cases in training (should be 0): ',pd.Series(train_cases).isin(valid_cases).sum())
        for i, ds in enumerate(sets):
            use_cases = list(target_df.loc[ds, 'case'])

            # Balance n tiles drawn per-target:
            n = []
            on = []
            if balance_tiles_per_target[i]:
                ntarget = ntargets[i]  # How many tiles to use in train/test per case (if re-balancing)
                for ii, target_id in enumerate(use_cases):
                    keep = np.argwhere(np.array(all_dat.loc[:, 'case'].isin([target_id]))
                                       ).flatten()
                    n0 = keep.shape[0]
                    on.append(n0)
                    if n0 > ntarget:
                        keep = resample(keep,
                                        n_samples=ntarget,
                                        random_state=fold,  # Re-balance differently on each fold
                                        replace=False, )
                    if ii == 0:
                        bal_dat = all_dat.loc[keep, ('fn', bal_target,'case', 'anno')].reset_index(drop=True)
                    else:
                        temp = all_dat.loc[keep, ('fn', bal_target,'case', 'anno')].reset_index(drop=True)
                        bal_dat = pd.concat((bal_dat, temp), axis=0)

                    ntiles = keep.shape[0]
                    n.append(ntiles)
                print('\t pre rebalance: median n tiles / case %s = %4.1f'
                      % (set_lab[i], np.median(on)))
                print('\t after rebalance: median n tiles / case %s = %4.1f'
                      % (set_lab[i], np.median(n)))
                bal_dat.reset_index(inplace=True, drop=True)
            else:
                # Initialize dataframe with all tiles from use_cases:
                use_tiles = all_dat.loc[:, 'case'].isin(
                    use_cases)  # Index of tiles from target(s) in current set 'ds' (train / valid)
                bal_dat = pd.DataFrame(all_dat.loc[use_tiles, ('fn', bal_target,'case', 'anno')])
                bal_dat = bal_dat.reset_index(drop=True)

            # Balance final classes from tiles in bal_dat:
            if balance_classes_total_tiles[i]:
                # Resample tiles to have equal number of class examples following any previous balancing:
                use_fn = bal_dat.loc[:, 'fn']
                
                use_class = bal_dat.loc[:, 'anno'].str.contains(pos_class)
                npos = np.sum(use_class)
                nneg = len(use_class) - npos
                use_n = np.min([npos, nneg])
                if use_n == 0: #If there are no targets in one class, revert to n of class with targets!
                    use_n = np.max([npos, nneg])
                print('Balancing %s tiles... %d neg %d pos, use %d'
                      % (set_lab[i], nneg, npos, use_n))
                pos = np.argwhere(np.array(use_class) == True)
                neg = np.argwhere(np.array(use_class) == False)
                if npos > nneg:
                    new = resample(pos,
                                   replace=False,
                                   n_samples=use_n,
                                   random_state=fold)
                    bal_tiles_idx = np.concatenate((neg, new))
                elif nneg > npos:
                    new = resample(neg,
                                   replace=False,
                                   n_samples=use_n,
                                   random_state=fold)
                    bal_tiles_idx = np.concatenate((new, pos))
                else:  # It can happen that they are already exactly equal!
                    bal_tiles_idx = np.concatenate((neg, pos))

                temp = pd.DataFrame(bal_dat.loc[bal_tiles_idx[:, 0],
                                                ('fn', bal_target,'case', 'anno')]
                                    ).reset_index(drop=True)
            else:
                use_class = bal_dat.loc[:, 'anno'].str.contains(pos_class).values
                npos = np.sum(use_class)
                nneg = len(use_class) - npos
                print('Not balancing class totals for %s ... %d neg %d pos, use all.'
                      % (set_lab[i], nneg, npos))
                temp = pd.DataFrame(bal_dat.loc[:, ('fn', bal_target, 'case','anno')]
                                    ).reset_index(drop=True)
            temp['is_valid'] = i
            if i == 0:
                out = temp
            else:
                out = pd.concat((out, temp), axis=0)

        print('if nbal > 0, n_pos =',
              np.sum(np.array(out.loc[out.loc[:, 'is_valid'] == 0, 'anno']) == pos_class),
              '(should ==) n_neg=',
              np.sum(np.array(out.loc[out.loc[:, 'is_valid'] == 0, 'anno']) != pos_class)
              )
        
        out.loc[:,'full_path'] = out.fn
        fn_only = [Path(fn).parts[-1] for fn in out.full_path]
        out.loc[:,'fn'] = fn_only
        print('Save to %s' % csv_path.joinpath('train_valid_fold_%d.csv' % fold))
        out.to_csv(csv_path.joinpath('train_valid_fold_%d.csv' % fold))
        fold += 1
elif fold_method == 'newdata':
    fold = 0
    out = all_dat.loc[:, ('fn', bal_target,'case', 'anno')]
    full_path_fn = [split_data_path.joinpath(fn) for fn in out['fn']]
    out['full_path'] = full_path_fn
    out['is_valid'] = 1  # All data are test / validation data
    fn = csv_path.joinpath('train_valid_fold_%d.csv' % fold)
    print('Save to %s' % fn)
    out.to_csv(fn)

    #  To extent indicated by use_model, create symlinks to this file:
    for i in range(1, (kfold * nrep)):
        newfn = csv_path.joinpath(csv_path.joinpath('train_valid_fold_%d.csv' % i))
        if newfn.exists():
            os.remove(str(newfn))
        os.symlink(str(fn), str(newfn))
