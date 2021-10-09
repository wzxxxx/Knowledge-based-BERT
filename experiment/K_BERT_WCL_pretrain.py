from experiment import build_data
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from experiment.my_nn import collate_pretrain_data, EarlyStopping, run_a_pretrain_epoch, \
    set_random_seed, K_BERT_WCL
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
set_random_seed()

# define parameters of model
args = {}
args['device'] = "cuda" if torch.cuda.is_available() else "cpu"
args['batch_size'] = 80
args['num_epochs'] = 50
args['d_model'] = 768
args['n_layers'] = 6
args['vocab_size'] = 47
args['maxlen'] = 201
args['d_k'] = 64
args['d_v'] = 64
args['d_ff'] = 768*4
args['n_heads'] = 12
args['global_labels_dim'] = 154
args['atom_labels_dim'] = 15
args['lr'] = 0.00003
args['task_name'] = 'k_bert_wcl'
args['pretrain_data_path'] = '../data/pretrain_data/CHEMBL_maccs'
pretrain_set = build_data.load_data_for_pretrain(
    pretrain_data_path=args['pretrain_data_path'])
print("Pretrain data generation is complete !")

pretrain_loader = DataLoader(dataset=pretrain_set,
                             batch_size=args['batch_size'],
                             shuffle=True,
                             collate_fn=collate_pretrain_data)

global_pos_weight = torch.tensor([884.17, 70.71, 43.32, 118.73, 428.67, 829.0, 192.84, 67.89, 533.86, 18.46, 707.55, 160.14, 23.19, 26.33, 13.38, 12.45, 44.91, 173.58, 40.14, 67.25, 171.12, 8.84, 8.36, 43.63, 5.87, 10.2, 3.06, 161.72, 101.75, 20.01, 4.35, 12.62, 331.79, 31.17, 23.19, 5.91, 53.58, 15.73, 10.75, 6.84, 3.92, 6.52, 6.33, 6.74, 24.7, 2.67, 6.64, 5.4, 6.71, 6.51, 1.35, 24.07, 5.2, 0.74, 4.78, 6.1, 62.43, 6.1, 12.57, 9.44, 3.33, 5.71, 4.67, 0.98, 8.2, 1.28, 9.13, 1.1, 1.03, 2.46, 2.95, 0.74, 6.24, 0.96, 1.72, 2.25, 2.16, 2.87, 1.8, 1.62, 0.76, 1.78, 1.74, 1.08, 0.65, 0.97, 0.71, 5.08, 0.75, 0.85, 3.3, 4.79, 1.72, 0.78, 1.46, 1.8, 2.97, 2.18, 0.61, 0.61, 1.83, 1.19, 4.68, 3.08, 2.83, 0.51, 0.77, 6.31, 0.47, 0.29, 0.58, 2.76, 1.48, 0.25, 1.33, 0.69, 1.03, 0.97, 3.27, 1.31, 1.22, 0.85, 1.75, 1.02, 1.13, 0.16, 1.02, 2.2, 1.72, 2.9, 0.26, 0.69, 0.6, 0.23, 0.76, 0.73, 0.47, 1.13, 0.48, 0.53, 0.72, 0.38, 0.35, 0.48, 0.12, 0.52, 0.15, 0.28, 0.36, 0.08, 0.06, 0.03, 0.07, 0.01])
atom_pos_weight = torch.tensor([4.81, 1.0, 2.23, 53.49, 211.94, 0.49, 2.1, 1.13, 1.22, 1.93, 5.74, 15.42, 70.09, 61.47, 23.2])
loss_criterion_global = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=global_pos_weight.to('cuda'))
loss_criterion_atom = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=atom_pos_weight.to('cuda'))
model = K_BERT_WCL(d_model=args['d_model'], n_layers=args['n_layers'], vocab_size=args['vocab_size'],
                   maxlen=args['maxlen'], d_k=args['d_k'], d_v=args['d_v'], n_heads=args['n_heads'], d_ff=args['d_ff'],
                   global_label_dim=args['global_labels_dim'], atom_label_dim=args['atom_labels_dim'])
optimizer = Adam(model.parameters(), lr=args['lr'])
stopper = EarlyStopping(task_name=args['task_name'])
model.to(args['device'])

for epoch in range(args['num_epochs']):
    start = time.time()
    # Train
    run_a_pretrain_epoch(args, epoch, model, pretrain_loader, loss_criterion_global=loss_criterion_global,
                         loss_criterion_atom=loss_criterion_atom, optimizer=optimizer)
    # Validation and early stop
    stopper.pretrain_step(epoch, model)
    elapsed = (time.time() - start)
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    print("An epoch time used:", "{:d}:{:d}:{:d}".format(int(h), int(m), int(s)))












