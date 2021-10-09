import pandas as pd
import numpy as np
from experiment.atom_embedding_generator import bert_atom_embedding
task_list = ['drugbank']
for task_name in task_list:
    print(task_name)
    dataset = pd.read_csv('./data/'+task_name+'_canonical.csv', index_col=None)
    smiles_list = dataset['canonical_smiles'].values.tolist()
    pretrain_features_list = []
    for i, smiles in enumerate(smiles_list):
        print("{}/{}".format(i+1, len(smiles_list)))
        try:
            h_global, g_atom = bert_atom_embedding(smiles, pretrain_model='pretrain_k_bert_epoch_7.pth')
            pretrain_features_list.append(h_global)
        except:
            pretrain_features_list.append(['NaN' for x in range(768)])

    for i in range(len(pretrain_features_list[0])):
        global_feature_n = [pretrain_features_list[x][i] for x in range(len(pretrain_features_list))]
        dataset['pretrain_feature_'+str(i+1)] = global_feature_n
    dataset = dataset[dataset['pretrain_feature_1']!='NaN']
    dataset.to_csv('./data/embedding/'+task_name+'_k_bert_embedding.csv', index=False)

