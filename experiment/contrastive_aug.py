from rdkit import Chem
import pandas as pd

def task_dataset_augmentation(task_name, input_path, output_path, augmentation_num=5):
    origin_dataset = pd.read_csv(input_path, index_col=None)
    smiles_list = origin_dataset['smiles'].values.tolist()

    for i in range(augmentation_num-1):
        aug_smiles = []
        for j, smiles in enumerate(smiles_list):
            print('{}/{}'.format(j + 1, len(smiles_list)))
            try:
                aug_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), doRandom=True))
            except:
                print(smiles)
                aug_smiles.append(smiles)
        origin_dataset['aug_smiles' + '_' + str(i)] = aug_smiles
    print(task_name)
    origin_dataset.to_csv(output_path, index=False)


task_name_list = ['Pgp-sub', 'HIA', 'F(20%)', 'F(30%)',  'FDAMDD', 'CYP1A2-sub', 'CYP2C19-sub', 'CYP2C9-sub',
                  'CYP2D6-sub', 'CYP3A4-sub', 'T12', 'DILI', 'SkinSen', 'Carcinogenicity', 'Respiratory']

aug_times = [5]

# DOWNSTREAM TASKS
for task_name in task_name_list:
    for times in aug_times:
        print(task_name)
        input_path = '../data/ADMETlab_data/' + task_name +'_canonical.csv'
        output_path = '../data/contrastive_data/' + task_name +'_'+str(times)+'_contrastive_aug.csv'
        task_dataset_augmentation(task_name, input_path, output_path, augmentation_num=times)

# PRETRAIN TASKS
task_dataset_augmentation('CHEMBL', input_path='../data/pretrain_data/CHEMBL.csv',
                          output_path='../data/pretrain_data/CHEMBL_contrastive.csv', augmentation_num=5)
