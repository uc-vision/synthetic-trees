# synthetic-trees

## Installation

``` pip install . ```

## Usage 

#### Downloading

To download the data, use `download.py` script. This can be called
using ``

Where: 
- `directory` is the directory of where you want to download to.

#### Visualising

To visualize the data, use the `visualize.py` script. This can be called using:
`view-synthetic-trees -p=file_path -d=directory, -lw=linewidth` 

Where:
- `file_path` is the path of the *.npz file of a single tree [optional].
- `directory` is the directory of folder containing *.npz files [optional].
- `linewidth` is the width of the skeleton lines in the visualizer [optional].

#### Evaluation

To evaluate the data from your method against the ground truth data, use the `evaluate.py`
scripts. This can be called using:
`evaluate-synthetic-trees -d_gt=ground_truth_directory -d_o=output_directory, -r_o=results_save_path` 

Where:
- `d_gt` is the directory of the ground truth *.npz files.
- `d_o` is the directory of folder containing your skeleton outputs (in *.ply) format.
- `ro` is the the *.csv path to save your results to.

#### Processing Results




## Installation

Synthetic tree dataset, links to data and scripts for processing and evaluation.

## Dataset

Links to data goes here.

## Usage

Instructions for processing/use as a library goes here.

` pip install . `
`conda install pytorch torchvision torchaudio pytorch-cuda=11.7 cudatoolkit=11.7 -c pytorch -c nvidia`

Follow instructions on FRNN github
