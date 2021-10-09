from experiment import build_data
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from experiment.my_nn import collate_pretrain_data, EarlyStopping, run_a_contrastive_R_S_pretrain_epoch, \
    set_random_seed, K_BERT
import time
set_random_seed()

# define parameters of model
args = {}
args['device'] = "cuda" if torch.cuda.is_available() else "cpu"
args['batch_size'] = 32
args['num_epochs'] = 50
args['d_model'] = 768
args['n_layers'] = 6
args['vocab_size'] = 47
args['maxlen'] = 201
args['d_k'] = 64
args['d_v'] = 64
args['d_ff'] = 768*4
args['n_heads'] = 12
args['global_labels_dim'] = 1
args['atom_labels_dim'] = 15
args['lr'] = 0.00003
args['pretrain_layer'] = 5
args['pretrain_model'] = 'pretrain_k_bert_epoch_7.pth'
args['task_name'] = 'k_bert_chirality_R_S'
args['pretrain_data_path'] = '../data/pretrain_data/chirality_pretrain_R_S_maccs'

pretrain_set = build_data.load_data_for_contrastive_aug_pretrain(
                                        pretrain_data_path=args['pretrain_data_path'])
print("Pretrain data generation is complete !")

pretrain_loader = DataLoader(dataset=pretrain_set,
                             batch_size=args['batch_size'],
                             shuffle=True,
                             collate_fn=collate_pretrain_data)

loss_criterion_global = torch.nn.BCEWithLogitsLoss(reduction='none')
loss_criterion_atom = torch.nn.BCEWithLogitsLoss(reduction='none')
model = K_BERT(d_model=args['d_model'], n_layers=args['n_layers'], vocab_size=args['vocab_size'],
               maxlen=args['maxlen'], d_k=args['d_k'], d_v=args['d_v'], n_heads=args['n_heads'], d_ff=args['d_ff'],
               global_label_dim=args['global_labels_dim'], atom_label_dim=args['atom_labels_dim'])
optimizer = Adam(model.parameters(), lr=args['lr'])
stopper = EarlyStopping(pretrained_model=args['pretrain_model'],
                        pretrain_layer=args['pretrain_layer'],
                        task_name=args['task_name'])
model.to(args['device'])
stopper.load_pretrained_model(model)

for epoch in range(args['num_epochs']):
    start = time.time()
    # Train
    run_a_contrastive_R_S_pretrain_epoch(args, epoch, model, pretrain_loader, loss_criterion_global=loss_criterion_global,
                                         loss_criterion_atom=loss_criterion_atom, optimizer=optimizer)
    # Validation and early stop
    stopper.pretrain_step(epoch, model)
    elapsed = (time.time() - start)
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    print("An epoch time used:", "{:d}:{:d}:{:d}".format(int(h), int(m), int(s)))












