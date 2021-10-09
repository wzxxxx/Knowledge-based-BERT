from experiment import build_data
task_list = ['Pgp-sub', 'HIA', 'F(20%)', 'F(30%)',  'FDAMDD', 'CYP1A2-sub', 'CYP2C19-sub', 'CYP2C9-sub',
             'CYP2D6-sub', 'CYP3A4-sub', 'T12', 'DILI', 'SkinSen', 'Carcinogenicity', 'Respiratory']
for task in task_list:
        build_data.built_data_and_save_for_splited(
                origin_path='../data/ADMETlab_data/' + task + '_canonical.csv',
                save_path='../data/task_data/' + task + '.npy')