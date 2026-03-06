#!/bin/bash
#SBATCH --job-name=MucTaL_preproc
#SBATCH --time=24:00:00 ## HH:MM:SS
#SBATCH --mem=256g
#SBATCH --partition=htc
#SBATCH --nodes=1

module load singularity
module load java
module load rclone

#########################################################
#########################################################
#Variables that will need to be updated for each project:
echo $SLURM_JOB_NODELIST
echo $SLURM_ARRAY_TASK_ID
BASE=$1
echo $BASE
RESULTS_PATH=$2 #Path where tiles / inference outputs will be stored
SCRIPT_PATH=$BASE"/scripts/pancan_pipe" #Path of git clone of this codebase
SAMPLES=$3 #path to .tsv file. $BASE/"sampleinfo/sample_list.tsv"
#Columns: task     slide         slide_path          anno_path [optional]
#          1     example.svs   /full/path/to/example.svs /full/path/to/example.geojson  
#[WARNING: make sure there are no spaces in file path!]

## Choose which parts of the pipeline to run:
RUN_THUMB=0
RUN_PREPROC=$5
KEEP_TILES=$6 # if =0 individual tiles will not be saved, only used as an intermediate for inference and generating outputs
JOB_OFFSET=$7 #i.e. if all jobs must stay below 500, use array 1-200 and add 200 each time
RUN_INFER=$8 #Requires preproc
RUN_POST=$9 #Requres preproc -> infer (creates heatmaps, geojson)
#Size of the tiles to use for inference:
TILE_SIZE=${10} #224 is default. Important to keep matched to model training size
USE_ANNO=${11}
echo 'USE_ANNO:' $USE_ANNO
TASK=$((SLURM_ARRAY_TASK_ID + JOB_OFFSET)) #$((num1 + num2))
echo 'tile size ' $TILE_SIZE 'px'
THUMB_WIDTH=1000 #Size of thumbnail outputs in pixels
MODEL_PATH=${12} #$SCRIPT_PATH/model/full_model.pkl
POS_CLASS=${13} #0 1 2 etc. default = 0
IGNORE_PROBLEMS=${14}
IGNORE_PROBLEMS=${IGNORE_PROBLEMS:-0}

#####################################################################
#####################################################################
### Do not change these variables unless you know what you're doing
## 
# 
#This currently does not work with filenames / paths with SPACES in them!!!
SLIDE=$(awk -v task=$TASK '$1==task {print $2}' $SAMPLES)
SLIDE_NAME=${SLIDE%.*} 
SLIDE_PATH=$(awk -v task=$TASK '$1==task {print $3}' $SAMPLES)

FILE_TYPE=".${SLIDE##*.}"
echo "File type:" $FILE_TYPE
#Copy the WSI file to scratch:
echo 'copy ' $SLIDE_PATH ' to ' $SLURM_SCRATCH
rclone copy $SLIDE_PATH $SLURM_SCRATCH --copy-links --progress #Updated to copy links
SLIDE_PATH=$SLURM_SCRATCH/$SLIDE

if [ $USE_ANNO = 1 ]
then
    ANNO_PATH=$(awk -v task=$TASK '$1==task {print $4}' $SAMPLES)
    echo $ANNO_PATH
    rclone copy $ANNO_PATH $SLURM_SCRATCH --copy-links --progress
    ANNO_PATH=$SLURM_SCRATCH/$SLIDE_NAME'.geojson'
else
    ANNO_PATH="NA"
fi

echo 'SLURM paths: ' 
echo $SLIDE_PATH
echo 'Annotations: ' $ANNO_PATH
echo 'observed'
ls $SLURM_SCRATCH
echo "Task ${TASK} using ${SLIDE}."

#############################################################
#### 1) First run H&E whole slide image tiling (This can take many hours depending on image size)
# JOB_SCRIPT=$SCRIPT_PATH/pipeline/test_pause.py
JOB_SCRIPT=$SCRIPT_PATH/pipeline/pathml_preproc.py
SING_IMG="/path/to/pathml_singularity_container/pathmlv.sif" 

# Processing image
# TODO: whether or not KEEP_TILES=1 should run on scratch! then copy back over at the end.
if [ $KEEP_TILES = 1 ]
then
    TILE_PATH=$RESULTS_PATH
else
    TILE_PATH=$SLURM_SCRATCH"/results/"
fi

if [ $RUN_PREPROC = 1 ]
then
    echo "Running proprocessing with ${JOB_SCRIPT}"
    echo "Tile output: ${TILE_PATH}"
    DATE_STR=$(date "+%Y-%m-%d_%H-%M-%S")
    echo $DATE_STR
    echo $SING_IMG ', ' $TILE_PATH ', ' $SLIDE_PATH ', ' $ANNO_PATH
    singularity exec $SING_IMG \
        /opt/conda/envs/py38/bin/python -u  $JOB_SCRIPT \
            $TILE_PATH \
            $SLIDE_PATH \
            $TILE_SIZE \
            $SCRIPT_PATH \
            $USE_ANNO \
            $ANNO_PATH \
            $IGNORE_PROBLEMS
    echo "Preprocessing complete."
fi
DATE_STR=$(date "+%Y-%m-%d_%H-%M-%S")
echo $DATE_STR