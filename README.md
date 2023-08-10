# <center> 🌳🌲🌴 Synthetic-Trees 🌴🌲🌳 </center>

## 📝 Description

This repository offers a synthetic point cloud dataset with ground truth skeletons of multiple species. Our library enables you to open, visualize, and assess the accuracy of the skeletons. To gain insight into how we created the data and the evaluation metrics employed, please refer to our published paper, available at this <a href="https://arxiv.org/abs/2303.11560">link</a>. Our dataset consists of two parts: one with point clouds that feature foliage, which is particularly useful for training models that can handle real-world data that includes leaves; the other contains only the branching structure and is less affected to occlusion.


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

## 🔍 Usage

#### 💾 Downloading

You can download the data by following this <a href="https://www.dropbox.com/sh/dkp3sgw6wpdiaam/AAAIRy8liOpy-y9jM6KCiNpNa?dl=0">link</a>. The dataset includes synthetic point clouds and ground truth skeletons, along with a JSON file that specifies the training, validation, and test sets. For evaluation purposes, we have provided "cleaned" point clouds and skeletons in the evaluation folder, which are suitable for assessment.


#### 🖥️ Installation 
To install:
Create a conda enviroment:

`conda create -n synthetic-trees python=3.8`

then:

`pip install .` 

#### 🕵️‍♂️ Visualizing 
To visualize the data, use the `visualize.py` script. You can call it using either: 

``` view-synthetic-trees -p=file_path -lw=linewidth ``` 
or 
``` view-synthetic-trees -d=directory -lw=linewidth ``` 

where: 
- `file_path` is the path of the `.npz` file of a single tree. 
- `directory` is the path of the folder containing `.npz` files.
- `linewidth` is the width of the skeleton lines in the visualizer. 

#### 📊 Evaluation 
To evaluate your method against the ground truth data, use the `evaluate.py` script. You can call it using: 

``` evaluate-synthetic-trees -d_gt=ground_truth_directory -d_o=output_directory -r_o=results_save_path ``` 

where: 
- `ground_truth_directory` is the directory of the ground truth `.npz` files. 
- `output_directory` is the directory of the folder containing your skeleton outputs (in `.ply` format).
- `results_save_path` is the path of the `.csv` file to save your results to. 

#### 📋 Processing Results
After running the evaluation, you can use the `process_results.py` script to post-process the raw results and obtain metrics across the dataset. Call it using: 

``` process-synthetic-trees-results -p=path ``` 

where: `path` is the path of the results `.csv` file from the evaluation step. 

## 📜 Citation 
Please use the following BibTeX entry to cite our work: <br>
```
@InProceedings{10.1007/978-3-031-36616-1_28,
author="Dobbs, Harry
and Batchelor, Oliver
and Green, Richard
and Atlas, James",
editor="Pertusa, Antonio
and Gallego, Antonio Javier
and S{\'a}nchez, Joan Andreu
and Domingues, In{\^e}s",
title="Smart-Tree: Neural Medial Axis Approximation of Point Clouds for 3D Tree Skeletonization",
booktitle="Pattern Recognition and Image Analysis",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="351--362",
abstract="This paper introduces Smart-Tree, a supervised method for approximating the medial axes of branch skeletons from a tree point cloud. Smart-Tree uses a sparse voxel convolutional neural network to extract the radius and direction towards the medial axis of each input point. A greedy algorithm performs robust skeletonization using the estimated medial axis. Our proposed method provides robustness to complex tree structures and improves fidelity when dealing with self-occlusions, complex geometry, touching branches, and varying point densities. We evaluate Smart-Tree using a multi-species synthetic tree dataset and perform qualitative analysis on a real-world tree point cloud. Our experimentation with synthetic and real-world datasets demonstrates the robustness of our approach over the current state-of-the-art method. The dataset (https://github.com/uc-vision/synthetic-trees) and source code (https://github.com/uc-vision/smart-tree) are publicly available.",
isbn="978-3-031-36616-1"
}

```

## 📥 Contact 

Should you have any questions, comments or suggestions please use the following contact details:
harry.dobbs@pg.canterbury.ac.nz
