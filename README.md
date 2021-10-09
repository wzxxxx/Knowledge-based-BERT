# Knowledge-based-BERT
 K-BERT is a model based on BERT that can extract molecular features from molecules like a computational chemist. The pre-training tasks are used in K-BERT: atom feature prediction task, global feature prediction task, and contrastive learning task. The atom feature prediction task allows the model to learn the manual extracted information in graph-based methods: atomic initial information, the global feature prediction task allows the model to learn the manual extracted information in descriptor-based methods: molecular descriptors/molecular fingerprints, and the contrastive learning task allows the model to make the embeddings of different SMILES strings of the same molecule more similar, thus enabling K-BERT to generalize to SMILES of different formats not limited to canonical SMILES.

![image](<https://github.com/wzxxxx/Knowledge-based-BERT/blob/main/figure/Knowledge-based%20BERT.png>)



**requirements：**
python 3.7
anaconda
xgboost
rdkit
pytorch
sklearn



The datasets and pre-trained models can be downloaded from the following link: https://pan.baidu.com/s/1yzhHwhELuJG-3lxlrVtRPA  Fetch code：WZXX

