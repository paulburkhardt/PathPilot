# Commands

## First access the Cluster 
```
salloc --nodes=1 --cpus-per-task=4 --mem=32G --gres=gpu:1,VRAM:24G --time=6:00:00 --mail-type=ALL  --part=PRACT --qos="practical_course"
```

## Activate the conda environment
```
conda activate mast3r-slam
```

## Make sure that the correct cuda version is loaded
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


## Run rerun

Forward the port 9876 to your local machine

Run rerun
```
rerun --serve-web
```

Open a new window -> in tmux for split screen

```
ctrl + b + %
```

Then run the "pipeline" in the other window

```
python demo_pathpilot.py
```

Visualize the results

```
rerun one_chair_pathpilot.rrd
```

The last recording is the one that is relevant.