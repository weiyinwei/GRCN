# Graph-Refined Convolutional Network for Multimedia Recommendation with Implicit Feedback
This is our Pytorch implementation for the paper:  
> Yinwei Wei, Xiang Wang, Liqiang Nie, Xiangnan He and Tat-Seng Chua. Graph-Refined Convolutional Network for Multimedia Recommendation with Implicit Feedback. In ACM MM`20, Seattle, United States, Oct. 12-16, 2020  
Author: Dr. Yinwei Wei (weiyinwei at hotmail.com)

## Introduction
In this work, we focus on adaptively refining the structure of interaction graph to discover and prune potential false-positive edges. Towards this end, we devise a new GCN-based recommendermodel, Graph-Refined Convolutional Network(GRCN), which adjusts the structure of interaction graph adaptively based on status of mode training, instead of remaining the fixed structure. 

## Environment Requirement
The code has been tested running under Python 3.5.2. The required packages are as follows:
- Pytorch == 1.4.0
- torch-cluster == 1.4.2
- torch-geometric == 1.2.1
- torch-scatter == 1.2.0
- torch-sparse == 0.4.0
- numpy == 1.16.0

## Example to Run the Codes
The instruction of commands has been clearly stated in the codes.
- Kwai dataset  
```python main.py --l_r=0.0001 --weight_decay=0.1 --dropout=0 --weight_mode=confid --num_routing=3 --is_pruning=False --data_path=Kwai --has_a=False --has_t=False```
- Tiktok dataset  
`python main.py --l_r=0.0001 --weight_decay=0.001 --dropout=0 --weight_mode=confid --num_routing=3 --is_pruning=False --data_path=Tiktok`
- Movielens dataset  
`python main.py --l_r=0.0001 --weight_decay=0.0001 --dropout=0 --weight_mode=confid --num_routing=3 --is_pruning=False`  

Some important arguments:  

- `weight_model` 
  It specifics the type of multimodal correlation integration. Here we provide three options:  
  1. `mean` implements the mean integration without confidence vectors. Usage `--weight_model 'mean'`
  2. `max` implements the max integration without confidence vectors. Usage `--weight_model 'max'`
  3. `confid` (by default)  implements the max integration with confidence vectors. Usage `--weight_model 'confid'`
  
- `fusion_mode` 
  It specifics the type of user and item representation in the prediction layer. Here we provide three options:  
  1. `concat` (by default) implements the concatenation of multimodal features. Usage `--fusion_mode 'concat'`
  2. `mean` implements the mean pooling of multimodal features. Usage `--fusion_mode 'max'`
  3. `id` implements the representation with only the id embeddings. Usage `--fusion_mode 'id'`
  

- `is_pruning` 
  It specifics the type of pruning operation. Here we provide three options:  
  1. `Ture` (by default) implements the hard pruning operations. Usage `--is_pruning 'True'`
  2. `False` implements the soft pruning operations. Usage `--is_pruning 'False'`
  
- 'has_v', 'has_a', and 'has_t' indicate the modality used in the model.

## Dataset
Please check [MMGCN](https://github.com/weiyinwei/MMGCN) for the datasets: Kwai, Tiktok, and Movielens. 

Due to the copyright, we could only provide some toy datasets for validation. If you need the complete ones, please contact the owners of the datasets. 
<!-- We follow [MMGCN](https://github.com/weiyinwei/MMGCN) and provide three processed datasets: Kwai, Tiktok, and Movielnes.  -->

||#Interactions|#Users|#Items|Visual|Acoustic|Textual|
|:-|:-|:-|:-|:-|:-|:-|
|Movielens|1,239,508|55,485|5,986|2,048|128|100|
|Tiktok|726,065|36,656|76,085|128|128|128|
|Kwai|298,492|86,483|7,010|2,048|-|-|

-`train.npy`
   Train file. Each line is a user with her/his positive interactions with items: (userID and micro-video ID)  
-`val.npy`
   Validation file. Each line is a user with her/his several positive interactions with items: (userID and micro-video ID)  
-`test.npy`
   Test file. Each line is a user with her/his several positive interactions with items: (userID and micro-video ID)  

