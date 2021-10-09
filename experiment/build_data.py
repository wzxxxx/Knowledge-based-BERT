import pandas as pdfrom rdkit.Chem import MolFromSmiles, MACCSkeys, AllChemimport numpy as npfrom rdkit.ML.Descriptors import MoleculeDescriptorsimport multiprocessing as mpimport torchfrom rdkit import Chemimport mathimport randomfrom rdkit.Chem import ChemicalFeaturesfrom rdkit import RDConfigimport os# knowledge-based transformer pre-train model# from rdkit_des import Chem# smi = ''# random_equivalent_smiles = Chem.MolFromSmiles(Chem.MolToSmiles(smi, doRandom=True))def smi_tokenizer(smi):    """    Tokenize a SMILES molecule or reaction    """    import re    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"    regex = re.compile(pattern)    tokens = [token for token in regex.findall(smi)]    # assert smi == ''.join(tokens)    # return ' '.join(tokens)    return tokensdef one_of_k_encoding(x, allowable_set):    if x not in allowable_set:        raise Exception("input {0} not in allowable set{1}:".format(            x, allowable_set))    return [x == s for s in allowable_set]def one_of_k_encoding_unk(x, allowable_set):    """Maps inputs not in the allowable set to the last element."""    if x not in allowable_set:        x = allowable_set[-1]    return [x == s for s in allowable_set]def atom_labels(atom, use_chirality=True):    results = one_of_k_encoding(atom.GetDegree(),                                [0, 1, 2, 3, 4, 5, 6]) + \              one_of_k_encoding_unk(atom.GetHybridization(), [                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,                  Chem.rdchem.HybridizationType.SP3D2, 'other']) + [atom.GetIsAromatic()] \              + one_of_k_encoding_unk(atom.GetTotalNumHs(),                                                  [0, 1, 2, 3, 4])    if use_chirality:        try:            results = results + one_of_k_encoding_unk(                atom.GetProp('_CIPCode'),                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]        except:            results = results + [False, False                                 ] + [atom.HasProp('_ChiralityPossible')]    atom_labels_list = np.array(results).tolist()    atom_selected_index = [1, 2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 17, 19, 20, 21]    atom_labels_selected = [atom_labels_list[x] for x in atom_selected_index]    return atom_labels_selecteddef global_maccs_data(smiles):    mol = Chem.MolFromSmiles(smiles)    maccs = MACCSkeys.GenMACCSKeys(mol)    global_maccs_list = np.array(maccs).tolist()    # 选择负/正样本比例小于1000且大于0.001的数据    selected_index = [3, 8, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165]    selected_global_list = [global_maccs_list[x] for x in selected_index]    return selected_global_listdef global_ecfp4_data(smiles):    mol = Chem.MolFromSmiles(smiles)    ecfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)    global_ecfp4_list = np.array(ecfp4).tolist()    return global_ecfp4_listdef global_rdkit_des_data(smiles):    descriptors_name = ['MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex', 'MinAbsEStateIndex', 'qed', 'MolWt',                           'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'NumRadicalElectrons',                           'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge',                           'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'BalabanJ', 'BertzCT', 'Chi0',                           'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n',                           'Chi4v', 'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1',                           'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2',                           'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9',                           'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6',                           'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11',                           'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6',                           'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'EState_VSA1', 'EState_VSA10',                           'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6',                           'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2',                           'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8',                           'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount',                           'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings',                           'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors',                           'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles',                           'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount', 'MolLogP', 'MolMR',                           'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH',                           'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine',                           'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2',                           'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide',                           'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide',                           'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo',                           'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido',                           'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan',                           'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy',                           'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho',                           'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol',                           'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine',                           'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd',                           'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan',                           'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']    m = Chem.MolFromSmiles(smiles)    desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptors_name)    descriptors = np.array(desc_calc.CalcDescriptors(m)).tolist()    return descriptorsdef construct_input_from_smiles(smiles, max_len=200, global_feature='MACCS'):    try:        # built a pretrain data from smiles        atom_list = []        atom_token_list = ['c', 'C', 'O', 'N', 'n', '[C@H]', 'F', '[C@@H]', 'S', 'Cl', '[nH]', 's', 'o', '[C@]',                           '[C@@]', '[O-]', '[N+]', 'Br', 'P', '[n+]', 'I', '[S+]',  '[N-]', '[Si]', 'B', '[Se]', '[other_atom]']        all_token_list = ['[PAD]', '[GLO]', 'c', 'C', '(', ')', 'O', '1', '2', '=', 'N', '3', 'n', '4', '[C@H]', 'F', '[C@@H]', '-', 'S', '/', 'Cl', '[nH]', 's', 'o', '5', '#', '[C@]', '[C@@]', '\\', '[O-]', '[N+]', 'Br', '6', 'P', '[n+]', '7', 'I', '[S+]', '8', '[N-]', '[Si]', 'B', '9', '[2H]', '[Se]', '[other_atom]', '[other_token]']        # 构建token转化成idx的字典        word2idx = {}        for i, w in enumerate(all_token_list):            word2idx[w] = i        # 构建token_list 并加上padding和global        token_list = smi_tokenizer(smiles)        padding_list = ['[PAD]' for x in range(max_len-len(token_list))]        tokens = ['[GLO]'] + token_list + padding_list        mol = MolFromSmiles(smiles)        atom_example = mol.GetAtomWithIdx(0)        atom_labels_example = atom_labels(atom_example)        atom_mask_labels = [2 for x in range(len(atom_labels_example))]        atom_labels_list = []        atom_mask_list = []        index = 0        tokens_idx = []        for i, token in enumerate(tokens):            if token in atom_token_list:                atom = mol.GetAtomWithIdx(index)                an_atom_labels = atom_labels(atom)                atom_labels_list.append(an_atom_labels)                atom_mask_list.append(1)                index = index + 1                tokens_idx.append(word2idx[token])            else:                if token in all_token_list:                    atom_labels_list.append(atom_mask_labels)                    tokens_idx.append(word2idx[token])                    atom_mask_list.append(0)                elif '[' in list(token):                    atom = mol.GetAtomWithIdx(index)                    tokens[i] = '[other_atom]'                    an_atom_labels = atom_labels(atom)                    atom_labels_list.append(an_atom_labels)                    atom_mask_list.append(1)                    index = index + 1                    tokens_idx.append(word2idx['[other_atom]'])                else:                    tokens[i] = '[other_token]'                    atom_labels_list.append(atom_mask_labels)                    tokens_idx.append(word2idx['[other_token]'])                    atom_mask_list.append(0)        if global_feature == 'MACCS':            global_label_list = global_maccs_data(smiles)        elif global_feature == 'ECFP4':            global_label_list = global_ecfp4_data(smiles)        elif global_feature == 'RDKIT_des':            global_label_list = global_rdkit_des_data(smiles)        tokens_idx = [word2idx[x] for x in tokens]        if len(tokens_idx) == max_len + 1:            return tokens_idx, global_label_list, atom_labels_list, atom_mask_list        else:            return 0, 0, 0, 0    except:        return 0, 0, 0, 0def build_maccs_pretrain_data_and_save(smiles_list, output_smiles_path, global_feature='MACCS'):    smiles_list = smiles_list    tokens_idx_list = []    global_label_list = []    atom_labels_list = []    atom_mask_list = []    for i, smiles in enumerate(smiles_list):        tokens_idx, global_labels, atom_labels, atom_mask = construct_input_from_smiles(smiles,                                                                                        global_feature=global_feature)        if tokens_idx != 0:            tokens_idx_list.append(tokens_idx)            global_label_list.append(global_labels)            atom_labels_list.append(atom_labels)            atom_mask_list.append(atom_mask)            print('{}/{} is transformed!'.format(i+1, len(smiles_list)))        else:            print('{} is transformed failed!'.format(smiles))    pretrain_data_list = [tokens_idx_list, global_label_list, atom_labels_list, atom_mask_list]    pretrain_data_np = np.array(pretrain_data_list)    np.save(output_smiles_path, pretrain_data_np)def build_ECFP4_pretrain_data_and_save(smiles_list, output_smiles_path, global_feature='ECFP4'):    smiles_list = smiles_list    tokens_idx_list = []    global_label_list = []    atom_labels_list = []    atom_mask_list = []    for i, smiles in enumerate(smiles_list):        tokens_idx, global_labels, atom_labels, atom_mask = construct_input_from_smiles(smiles,                                                                                        global_feature=global_feature)        if tokens_idx != 0:            tokens_idx_list.append(tokens_idx)            global_label_list.append(global_labels)            atom_labels_list.append(atom_labels)            atom_mask_list.append(atom_mask)            print('{}/{} is transformed!'.format(i+1, len(smiles_list)))        else:            print('{} is transformed failed!'.format(smiles))    pretrain_data_list = [tokens_idx_list, global_label_list, atom_labels_list, atom_mask_list]    pretrain_data_np = np.array(pretrain_data_list)    np.save(output_smiles_path, pretrain_data_np)def build_rdkit_des_pretrain_data_and_save(smiles_list, output_smiles_path, global_feature='RDKIT_des'):    smiles_list = smiles_list    tokens_idx_list = []    global_label_list = []    atom_labels_list = []    atom_mask_list = []    for i, smiles in enumerate(smiles_list):        tokens_idx, global_labels, atom_labels, atom_mask = construct_input_from_smiles(smiles,                                                                                        global_feature=global_feature)        if tokens_idx != 0:            tokens_idx_list.append(tokens_idx)            global_label_list.append(global_labels)            atom_labels_list.append(atom_labels)            atom_mask_list.append(atom_mask)            print('{}/{} is transformed!'.format(i+1, len(smiles_list)))        else:            print('{} is transformed failed!'.format(smiles))    pretrain_data_list = [tokens_idx_list, global_label_list, atom_labels_list, atom_mask_list]    pretrain_data_np = np.array(pretrain_data_list)    np.save(output_smiles_path, pretrain_data_np)def build_chirality_pretrain_data_and_save(smiles_list, labels_list, output_smiles_path):    tokens_idx_list = []    global_label_list = []    atom_labels_list = []    atom_mask_list = []    for i, smiles in enumerate(smiles_list):        tokens_idx, _, atom_labels, atom_mask = construct_input_from_smiles(smiles)        if tokens_idx != 0:            tokens_idx_list.append(tokens_idx)            global_label_list.append([labels_list[i]])            atom_labels_list.append(atom_labels)            atom_mask_list.append(atom_mask)            print('{}/{} is transformed!'.format(i+1, len(smiles_list)))        else:            print('{} is transformed failed!'.format(smiles))    pretrain_data_list = [tokens_idx_list, global_label_list, atom_labels_list, atom_mask_list]    pretrain_data_np = np.array(pretrain_data_list)    np.save(output_smiles_path, pretrain_data_np)def build_mask(labels_list, mask_value=100):    mask = []    for i in labels_list:        if i == mask_value:            mask.append(0)        else:            mask.append(1)    return maskdef multi_task_build_dataset(dataset_smiles, labels_list, smiles_name, global_feature='ECFP4'):    dataset = []    failed_molecule = []    labels = dataset_smiles[labels_list]    split_index = dataset_smiles['group']    smilesList = dataset_smiles[smiles_name]    molecule_number = len(smilesList)    for i, smiles in enumerate(smilesList):        token_idx, _, _, _ = construct_input_from_smiles(smiles, global_feature=global_feature)        if token_idx != 0:            mask = build_mask(labels.loc[i], mask_value=123456)            molecule = [smiles, token_idx, labels.loc[i].values.tolist(), mask, split_index.loc[i]]            dataset.append(molecule)            print('{}/{} molecule is transformed! {} is transformed failed!'.format(i + 1, molecule_number,                                                                                    len(failed_molecule)))        else:            print('{} is transformed failed!'.format(smiles))            molecule_number = molecule_number - 1            failed_molecule.append(smiles)    print('{}({}) is transformed failed!'.format(failed_molecule, len(failed_molecule)))    return datasetdef built_data_and_save_for_splited(        origin_path='G:/加密/Dataset/AttentionFP/ClinTox.csv',        save_path='G:/加密/Dataset/AttentionFP/ClinTox.npy',        task_list_selected=None):    data_origin = pd.read_csv(origin_path)    data_origin = data_origin.fillna(123456)    labels_list = [x for x in data_origin.columns if x not in ['smiles', 'group']]    if task_list_selected is not None:        labels_list = task_list_selected    data_set_gnn = multi_task_build_dataset(dataset_smiles=data_origin, labels_list=labels_list,                                            smiles_name='smiles')    smiles, token_idx, labels, mask, split_index = map(list, zip(*data_set_gnn))    dataset_list = [smiles, token_idx, labels, mask, split_index]    dataset_np = np.array(dataset_list)    np.save(save_path, dataset_np)    print('Molecules graph is saved!')def built_ECFP4_data_and_save_for_splited(        origin_path='G:/加密/Dataset/AttentionFP/ClinTox.csv',        save_path='G:/加密/Dataset/AttentionFP/ClinTox.npy',        task_list_selected=None):    data_origin = pd.read_csv(origin_path)    data_origin = data_origin.fillna(123456)    labels_list = [x for x in data_origin.columns if x not in ['smiles', 'group']]    if task_list_selected is not None:        labels_list = task_list_selected    data_set_gnn = multi_task_build_dataset(dataset_smiles=data_origin, labels_list=labels_list,                                            smiles_name='smiles', global_feature='ECFP4')    smiles, token_idx, labels, mask, split_index = map(list, zip(*data_set_gnn))    dataset_list = [smiles, token_idx, labels, mask, split_index]    dataset_np = np.array(dataset_list)    np.save(save_path, dataset_np)    print('Molecules graph is saved!')def built_rdkit_des_data_and_save_for_splited(        origin_path='G:/加密/Dataset/AttentionFP/ClinTox.csv',        save_path='G:/加密/Dataset/AttentionFP/ClinTox.npy',        task_list_selected=None):    data_origin = pd.read_csv(origin_path)    data_origin = data_origin.fillna(123456)    labels_list = [x for x in data_origin.columns if x not in ['smiles', 'group']]    if task_list_selected is not None:        labels_list = task_list_selected    data_set_gnn = multi_task_build_dataset(dataset_smiles=data_origin, labels_list=labels_list,                                            smiles_name='smiles', global_feature='RDKIT_des')    smiles, token_idx, labels, mask, split_index = map(list, zip(*data_set_gnn))    dataset_list = [smiles, token_idx, labels, mask, split_index]    dataset_np = np.array(dataset_list)    np.save(save_path, dataset_np)    print('Molecules graph is saved!')def contrastive_aug_build_dataset(dataset_smiles, labels_list, smiles_name_list):    dataset = []    failed_molecule = []    labels = dataset_smiles[labels_list]    split_index = dataset_smiles['group']    smilesList = dataset_smiles[smiles_name_list].values.tolist()    molecule_number = len(smilesList)    for i, _ in enumerate(smilesList):        token_idx_list = [construct_input_from_smiles(smiles)[0] for smiles in smilesList[i]]        if 0 not in token_idx_list:            mask = build_mask(labels.loc[i], mask_value=123456)            molecule = [smilesList[i][0], labels.loc[i].values.tolist(), mask, split_index.loc[i], token_idx_list]            dataset.append(molecule)            print('{}/{} molecule is transformed! {} is transformed failed!'.format(i + 1, molecule_number,                                                                                    len(failed_molecule)))        else:            print('{} is transformed failed!'.format(smilesList[i][0]))            molecule_number = molecule_number - 1            failed_molecule.append(smilesList[i][0])    print('{}({}) is transformed failed!'.format(failed_molecule, len(failed_molecule)))    return datasetdef built_data_and_save_for_contrastive_splited(        origin_path='G:/加密/Dataset/AttentionFP/ClinTox.csv',        save_path='G:/加密/Dataset/AttentionFP/ClinTox.npy'):    data_origin = pd.read_csv(origin_path)    data_origin = data_origin.fillna(123456)    smiles_list = ['smiles', 'aug_smiles_0', 'aug_smiles_1', 'aug_smiles_2', 'aug_smiles_3']    labels_list = [x for x in data_origin.columns if x not in ['smiles', 'group']+smiles_list ]    data_set = contrastive_aug_build_dataset(dataset_smiles=data_origin, labels_list=labels_list,                                             smiles_name_list=smiles_list)    smiles, labels, mask, split_index, token_idx, = map(list, zip(*data_set))    dataset_list = [smiles, token_idx, labels, mask, split_index]    dataset_np = np.array(dataset_list)    np.save(save_path, dataset_np)    print('Molecules graph is saved!')def build_maccs_pretrain_contrastive_data_and_save(smiles_list, output_smiles_path, global_feature='MACCS'):    # all smiles list    smiles_list = smiles_list    tokens_idx_all_list = []    global_label_list = []    atom_labels_list = []    atom_mask_list = []    for i, smiles_one_mol in enumerate(smiles_list):        tokens_idx_list = [construct_input_from_smiles(smiles, global_feature=global_feature)[0] for                           smiles in smiles_one_mol]        if 0 not in tokens_idx_list:            _ , global_labels, atom_labels, atom_mask = construct_input_from_smiles(smiles_one_mol[0],                                                                global_feature=global_feature)            tokens_idx_all_list.append(tokens_idx_list)            global_label_list.append(global_labels)            atom_labels_list.append(atom_labels)            atom_mask_list.append(atom_mask)            print('{}/{} is transformed!'.format(i+1, len(smiles_list)))        else:            print('{} is transformed failed!'.format(smiles_one_mol[0]))    pretrain_data_list = [tokens_idx_all_list, global_label_list, atom_labels_list, atom_mask_list]    pretrain_data_np = np.array(pretrain_data_list, dtype=object)    np.save(output_smiles_path, pretrain_data_np)def build_pretrain_chirality_R_S_contrastive_data_and_save(smiles_list, global_all_label_list, output_smiles_path, global_feature='MACCS'):    # all smiles list    smiles_list = smiles_list    tokens_idx_all_list = []    global_label_list = []    atom_labels_list = []    atom_mask_list = []    for i, smiles_one_mol in enumerate(smiles_list):        tokens_idx_list = [construct_input_from_smiles(smiles, global_feature=global_feature)[0] for                           smiles in smiles_one_mol]        if 0 not in tokens_idx_list:            _ , global_labels, atom_labels, atom_mask = construct_input_from_smiles(smiles_one_mol[0],                                                                global_feature=global_feature)            tokens_idx_all_list.append(tokens_idx_list)            global_label_list.append(global_all_label_list[i])            atom_labels_list.append(atom_labels)            atom_mask_list.append(atom_mask)            print('{}/{} is transformed!'.format(i+1, len(smiles_list)))        else:            print('{} is transformed failed!'.format(smiles_one_mol[0]))    pretrain_data_list = [tokens_idx_all_list, global_label_list, atom_labels_list, atom_mask_list]    pretrain_data_np = np.array(pretrain_data_list, dtype=object)    np.save(output_smiles_path, pretrain_data_np)def load_data_for_pretrain(pretrain_data_path='./data/CHEMBL_wash_500_pretrain'):    tokens_idx_list = []    global_labels_list = []    atom_labels_list = []    atom_mask_list = []    for i in range(80):        pretrain_data = np.load(pretrain_data_path+'_{}.npy'.format(i+1), allow_pickle=True)        tokens_idx_list = tokens_idx_list + [x for x in pretrain_data[0]]        global_labels_list = global_labels_list + [x for x in pretrain_data[1]]        atom_labels_list = atom_labels_list + [x for x in pretrain_data[2]]        atom_mask_list = atom_mask_list + [x for x in pretrain_data[3]]        print(pretrain_data_path+'_{}.npy'.format(i+1) + ' is loaded')    pretrain_data_final = []    for i in range(len(tokens_idx_list)):        a_pretrain_data = [tokens_idx_list[i], global_labels_list[i], atom_labels_list[i], atom_mask_list[i]]        pretrain_data_final.append(a_pretrain_data)    return pretrain_data_finaldef load_data_for_contrastive_aug_pretrain(pretrain_data_path='./data/CHEMBL_wash_500_pretrain'):    tokens_idx_list = []    global_labels_list = []    atom_labels_list = []    atom_mask_list = []    for i in range(80):        pretrain_data = np.load(pretrain_data_path+'_contrastive_{}.npy'.format(i+1), allow_pickle=True)        tokens_idx_list = tokens_idx_list + [x for x in pretrain_data[0]]        global_labels_list = global_labels_list + [x for x in pretrain_data[1]]        atom_labels_list = atom_labels_list + [x for x in pretrain_data[2]]        atom_mask_list = atom_mask_list + [x for x in pretrain_data[3]]        print(pretrain_data_path+'_contrastive_{}.npy'.format(i+1) + ' is loaded')    pretrain_data_final = []    for i in range(len(tokens_idx_list)):        a_pretrain_data = [tokens_idx_list[i], global_labels_list[i], atom_labels_list[i], atom_mask_list[i]]        pretrain_data_final.append(a_pretrain_data)    return pretrain_data_finaldef load_data_for_pretrain_rdkit_des(pretrain_data_path='./data/CHEMBL_wash_500_pretrain'):    tokens_idx_list = []    global_labels_list = []    atom_labels_list = []    atom_mask_list = []    for i in range(80):        pretrain_data = np.load(pretrain_data_path+'_{}.npy'.format(i+1), allow_pickle=True)        tokens_idx_list = tokens_idx_list + [x for x in pretrain_data[0]]        global_labels_list = global_labels_list + [x for x in pretrain_data[1]]        atom_labels_list = atom_labels_list + [x for x in pretrain_data[2]]        atom_mask_list = atom_mask_list + [x for x in pretrain_data[3]]        print(pretrain_data_path+'_{}.npy'.format(i+1) + ' is loaded')    global_labels_pd = pd.DataFrame(global_labels_list)    global_labels_normal = global_labels_pd.apply(lambda x: (x - x.mean()) / math.sqrt(sum((x - x.min()) ** 2 / len(x))))    global_labels_normal_final = global_labels_normal.dropna(axis=1, how='any')    pretrain_data_final = []    for i in range(len(tokens_idx_list)):        a_pretrain_data = [tokens_idx_list[i], global_labels_normal_final.iloc[i].values.tolist(), atom_labels_list[i], atom_mask_list[i]]        pretrain_data_final.append(a_pretrain_data)    global_labels_dim = len(global_labels_normal_final.iloc[1].values.tolist())    return pretrain_data_final, global_labels_dimdef load_data_for_augmentation_pretrain(pretrain_data_path='./data/CHEMBL_wash_500_pretrain'):    tokens_idx_list = []    global_labels_list = []    atom_labels_list = []    atom_mask_list = []    for i in range(40):        pretrain_data = np.load(pretrain_data_path+'_{}.npy'.format(i+1), allow_pickle=True)        tokens_idx_list = tokens_idx_list + [x for x in pretrain_data[0]]        global_labels_list = global_labels_list + [x for x in pretrain_data[1]]        atom_labels_list = atom_labels_list + [x for x in pretrain_data[2]]        atom_mask_list = atom_mask_list + [x for x in pretrain_data[3]]        print(pretrain_data_path+'_{}.npy'.format(i+1) + ' is loaded')    pretrain_data_final = []    for i in range(len(tokens_idx_list)):        a_pretrain_data = [tokens_idx_list[i], global_labels_list[i], atom_labels_list[i], atom_mask_list[i]]        pretrain_data_final.append(a_pretrain_data)    return pretrain_data_finaldef load_data_for_splited(data_path='example.npy'):    data = np.load(data_path, allow_pickle=True)    smiles_list = data[0]    tokens_idx_list = data[1]    labels_list = data[2]    mask_list = data[3]    group_list = data[4]    train_set = []    val_set = []    test_set = []    task_number = len(labels_list[1])    for i, group in enumerate(group_list):        molecule = [smiles_list[i], tokens_idx_list[i], labels_list[i], mask_list[i]]        if group == 'training':            train_set.append(molecule)        elif group == 'val':            val_set.append(molecule)        else:            test_set.append(molecule)    print('Training set: {}, Validation set: {}, Test set: {}, task number: {}'.format(            len(train_set), len(val_set), len(test_set), task_number))    return train_set, val_set, test_set, task_numberdef load_data_for_random_splited(data_path='example.npy', shuffle=True):    data = np.load(data_path, allow_pickle=True)    smiles_list = data[0]    tokens_idx_list = data[1]    labels_list = data[2]    mask_list = data[3]    group_list = data[4]    if shuffle:        random.shuffle(group_list)    print(group_list)    train_set = []    val_set = []    test_set = []    task_number = len(labels_list[1])    for i, group in enumerate(group_list):        molecule = [smiles_list[i], tokens_idx_list[i], labels_list[i], mask_list[i]]        if group == 'training':            train_set.append(molecule)        elif group == 'val':            val_set.append(molecule)        else:            test_set.append(molecule)    print('Training set: {}, Validation set: {}, Test set: {}, task number: {}'.format(            len(train_set), len(val_set), len(test_set), task_number))    return train_set, val_set, test_set, task_numberdef task_dataset_analyze(data_path='example.npy'):    data = np.load(data_path, allow_pickle=True)    smiles_list = data[0]    tokens_idx_list = data[1]    labels_list = data[2]    mask_list = data[3]    group_list = data[4]    train_set_pad = 0    val_set_pad = 0    test_set_pad = 0    train_set_other_atom = 0    val_set_other_atom = 0    test_set_other_atom = 0    train_set_other_token = 0    val_set_other_token = 0    test_set_other_token = 0    train_set = []    val_set = []    test_set = []    task_number = len(labels_list[1])    for i, group in enumerate(group_list):        tokens_idx_np = np.array(tokens_idx_list[i])        pad_count = len(np.where(tokens_idx_np == 0)[0])        other_atom_count = len(np.where(tokens_idx_np == 45)[0])        other_token_count = len(np.where(tokens_idx_np == 46)[0])        if group == 'training':            train_set.append(tokens_idx_np)            train_set_pad = train_set_pad + pad_count            train_set_other_atom = train_set_other_atom + other_atom_count            train_set_other_token = train_set_other_token + other_token_count        elif group == 'val':            val_set.append(tokens_idx_np)            val_set_pad = val_set_pad + pad_count            val_set_other_atom = val_set_other_atom + other_atom_count            val_set_other_token = val_set_other_token + other_token_count        else:            test_set.append(tokens_idx_np)            test_set_pad = test_set_pad + pad_count            test_set_other_atom = test_set_other_atom + other_atom_count            test_set_other_token = test_set_other_token + other_token_count    print('Training set, mol count: {},  pad count: {} {}%, other atom count: {}, other token count: {}'.format(            len(train_set), train_set_pad, round(train_set_pad/(len(train_set)*201)*100, 2), train_set_other_atom, train_set_other_token))    print('Validation set, mol count: {},  pad count: {} {}%, other atom count: {}, other token count: {}'.format(            len(val_set), val_set_pad, round(val_set_pad/(len(val_set)*201)*100, 2), val_set_other_atom, val_set_other_token))    print('Test set, mol count: {},  pad count: {} {}%, other atom count: {}, other token count: {}'.format(            len(test_set), test_set_pad, round(test_set_pad/(len(test_set)*201)*100, 2), test_set_other_atom, test_set_other_token))