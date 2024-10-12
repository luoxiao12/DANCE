# [NeurIPS 2024] Semi-supervised Knowledge Transfer Across Multi-omic Single-cell Data

## Introduction
DANCE is a semi-supervised framework for knowledge transfer across multi-omic single-cell data.

## Getting Started
#### Requirements
- Python 3.10, PyTorch>=1.21.0,  numpy>=1.24.0, are required for the current codebase.

#### Datasets
##### CITE-seq and ASAP-seq Data 
Download dataset from https://github.com/SydneyBioX/scJoint/blob/main/data.zip.

#### Cell Type Transfer 
<pre>python main.py --dataset cite-asap --label_ratio 0.1 </pre> 
 

## Acknowledgement
Our codebase is built based on [Transfer Learning Libary](https://github.com/thuml/Transfer-Learning-Library). We thank the authors for the nicely organized code!
