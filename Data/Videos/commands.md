# Commands

## First access the Cluster 
```
salloc --nodes=1 --cpus-per-task=4 --mem=32G --gres=gpu:1,VRAM:24G --time=6:00:00 --mail-type=ALL  --part=PRACT --qos="practical_course"
```

## Activate the conda environment
```
conda activate mast3r-slam
```

## Make sur that the correct cuda version is loaded
check cuda version and pytorch version
```
nvidia-smi
```
check cuda version
```
nvcc --version
```
load the correct cuda version
```
module load cuda/12.4.1
```




## Run on video from this folder
```
python main.py --dataset ../Data/Videos/one_chair.mp4 --config config/base.yaml --no-viz --save-as ../../plots/one_chair
```
