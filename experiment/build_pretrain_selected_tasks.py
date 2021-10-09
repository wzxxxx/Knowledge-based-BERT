from experiment.build_data import build_maccs_pretrain_data_and_save
import multiprocessing
import pandas as pd

task_name = 'CHEMBL'
if __name__ == "__main__":
    n_thread = 8
    data = pd.read_csv('../pretrain_data/'+task_name+'.csv')
    smiles_list = data['smiles'].values.tolist()
    # 避免内存不足，将数据集分为10份来计算
    for i in range(10):
        n_split = int(len(smiles_list)/10)
        smiles_split = smiles_list[i*n_split:(i+1)*n_split]

        n_mol = int(len(smiles_split)/8)

        # creating processes
        p1 = multiprocessing.Process(target=build_maccs_pretrain_data_and_save, args=(smiles_split[:n_mol],
                                                                                '../data/pretrain_data/'+task_name+'_maccs_'+str(i*8+1)+'.npy'))
        p2 = multiprocessing.Process(target=build_maccs_pretrain_data_and_save, args=(smiles_split[n_mol:2*n_mol],
                                                                                '../data/pretrain_data/'+task_name+'_maccs_'+str(i*8+2)+'.npy'))
        p3 = multiprocessing.Process(target=build_maccs_pretrain_data_and_save, args=(smiles_split[2*n_mol:3*n_mol],
                                                                                '../data/pretrain_data/'+task_name+'_maccs_'+str(i*8+3)+'.npy'))
        p4 = multiprocessing.Process(target=build_maccs_pretrain_data_and_save, args=(smiles_split[3*n_mol:4*n_mol],
                                                                                '../data/pretrain_data/'+task_name+'_maccs_'+str(i*8+4)+'.npy'))
        p5 = multiprocessing.Process(target=build_maccs_pretrain_data_and_save, args=(smiles_split[4*n_mol:5*n_mol],
                                                                                '../data/pretrain_data/'+task_name+'_maccs_'+str(i*8+5)+'.npy'))
        p6 = multiprocessing.Process(target=build_maccs_pretrain_data_and_save, args=(smiles_split[5*n_mol:6*n_mol],
                                                                                '../data/pretrain_data/'+task_name+'_maccs_'+str(i*8+6)+'.npy'))
        p7 = multiprocessing.Process(target=build_maccs_pretrain_data_and_save, args=(smiles_split[6*n_mol:7*n_mol],
                                                                                '../data/pretrain_data/'+task_name+'_maccs_'+str(i*8+7)+'.npy'))
        p8 = multiprocessing.Process(target=build_maccs_pretrain_data_and_save, args=(smiles_split[7*n_mol:],
                                                                                '../data/pretrain_data/'+task_name+'_maccs_'+str(i*8+8)+'.npy'))

        # starting my_scaffold_split 1&2
        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p5.start()
        p6.start()
        p7.start()
        p8.start()

        # wait until my_scaffold_split 1&2 is finished
        p1.join()
        p2.join()
        p3.join()
        p4.join()
        p5.join()
        p6.join()
        p7.join()
        p8.join()


        # both processes finished
        print("Done!")

