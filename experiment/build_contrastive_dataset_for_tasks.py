from experiment import build_data
task_name_list = ['Pgp-sub', 'HIA', 'F(20%)', 'F(30%)',  'FDAMDD', 'CYP1A2-sub', 'CYP2C19-sub', 'CYP2C9-sub',
                  'CYP2D6-sub', 'CYP3A4-sub', 'T12', 'DILI', 'SkinSen', 'Carcinogenicity', 'Respiratory']
aug_times = [5]
for task_name in task_name_list:
    for times in aug_times:
        build_data.built_data_and_save_for_contrastive_splited(
                origin_path='../data/contrastive_data/' + task_name + '_'+str(times) + '_contrastive_aug.csv',
                save_path='../data/task_data/'+ task_name + '_'+str(times) + '_contrastive_aug.npy')