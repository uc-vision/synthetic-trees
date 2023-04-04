# synthetic-trees

## Description

We provide a multi-species synthetic dataset with ground truth skeletons.
This repository contains a library to open, visualize and evaluate skeletons.
To understand how the data was created, aswell as the evaluation metrics used - please refer to our <a href="">paper</a>:

<table>
<tr>
  <td style="text-align: center"><img src="images/cherry-pcd.png", height=100%></td>
  <td style="text-align: center"><img src="images/apple-pcd.png", height=100%></td>
  <td style="text-align: center"><img src="images/pine-pcd.png", height=100%></td>
</tr>
<tr>
  <td align="center">Sapling Cherry Point Cloud.</td>
  <td align="center">Apple Tree Point Cloud.</td>
  <td align="center">Pine Tree Point Cloud.</td>
</tr>
  
<tr>
<td style="text-align: center"><img src="images/cherry-skeleton.png", height=100%></td>
<td style="text-align: center"><img src="images/apple-skeleton.png", height=100%></td>
<td style="text-align: center"><img src="images/pine-skeleton.png", height=100%></td>
</tr>
<tr>
  <td align="center">Sapling Cherry Ground Truth Skeleton.</td>
  <td align="center">Apple Tree Ground Truth Skeleton.</td>
  <td align="center">Pine Tree Ground Truth Skeleton.</td>
</tr>

</table>

## Usage

#### Downloading

The data can be downloaded from this <a href="https://www.dropbox.com/sh/dkp3sgw6wpdiaam/AAAIRy8liOpy-y9jM6KCiNpNa?dl=0">link</a>. <br>
There is a json file defining the train, validation and test split. <br>
The dataset contains the synthetic point clouds and ground truth skeletons. <br>
The `evaluation` folder contains 'cleaned' point clouds and skeleton that are suitable for evaluation.

#### Installation

`pip install .`


#### Visualising

To visualize the data, use the `visualize.py` script. This can be called using:

```
view-synthetic-trees -p=file_path -lw=linewidth
```

```
view-synthetic-trees -d=directory -lw=linewidth
```

Where:

- `file_path` is the path of the \*.npz file of a single tree [optional].
- `directory` is the directory of folder containing \*.npz files [optional].
- `linewidth` is the width of the skeleton lines in the visualizer [optional].

#### Evaluation

To evaluate the data from your method against the ground truth data, use the `evaluate.py`
script. This can be called using:

```
evaluate-synthetic-trees -d_gt=ground_truth_directory -d_o=output_directory, -r_o=results_save_path
```

Where:

- `ground_truth_directory` is the directory of the ground truth \*.npz files.
- `output_directory` is the directory of folder containing your skeleton outputs (in \*.ply) format.
- `results_save_path` is the the \*.csv path to save your results to.

#### Process Results

After running the evaluation, the raw results can be post-processed to provide metrics across the dataset.
This is done using the `process_results.py` script. This can be called using:

```
process-synthetic-trees-results -p=path
```

Where:

- `path` is the path of the results csv from the evaluation step.

## Citation
```
@inproceedings{TODO,
    author = {TODO},
     title = {{TODO}},
 booktitle = {TODO},
     pages = {TODO},
      year = {TODO}}
```

