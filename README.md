# [NeurIPS 2024] Semi-supervised Knowledge Transfer Across Multi-omic Single-cell Data

## Introduction
DANCE is a semi-supervised framework for knowledge transfer across multi-omic single-cell data.

![image](https://github.com/zfkarl/DANCE/tree/master/imgs/NeurIPS24-DANCE.png)

## Getting Started
#### Requirements
- torch>=1.7.0, torchvision>=0.5.0, numpy, prettytable, tqdm, scikit-learn, webcolors, matplotlib, opencv-python, numba are required for the current codebase.

#### Datasets
##### CITE-ASAP Data 
Download dataset from https://github.com/SydneyBioX/scJoint/blob/main/data.zip.

#### Cell Type Transfer 
<pre>python main.py --dataset cite-asap --label_ratio 0.1 </pre> 
 

## Acknowledgement
Our codebase is built based on [Transfer Learning Libary](https://github.com/thuml/Transfer-Learning-Library). We thank the authors for the nicely organized code!
