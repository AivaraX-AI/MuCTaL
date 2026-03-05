import os
import sys
import json
import math
import time
import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# 
#Begin the job
samples_fn = Path(sys.argv[1]) # version10 -> slide to use
dest = Path(sys.argv[2]) # e.g. /slurm/scratch12312
helper_path = sys.argv[3] #'/ix/rbao/Projects/panCancer_HE/scripts/pancancer_he_classifier'
move_column = sys.argv[4]
task = int(sys.argv[5])
sys.path.append(helper_path)
from helpers import file_handling as fileHelper

samples = pd.read_csv(samples_fn,sep = '\t')
fn = samples.loc[samples.task.values == task, move_column]
new_fn = fileHelper.remove_space_fn(fn)
new_dest = dest.joinpath(new_fn)
print(str(new_dest))
status = fileHelper.rclone_copy(fn,new_dest)
 
    
