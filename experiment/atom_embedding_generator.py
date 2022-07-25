https://github.com/wzxxxx/Knowledge-based-BERT/tree/main/experimentfrom experiment.build_data import construct_input_from_smiles
import torch
from experiment.my_nn import EarlyStopping, set_random_seed, BERT_atom_embedding_generator
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
set_random_seed()


def bert_atom_embedding(smiles, pretrain_model='pretrain_k_bert_epoch_7.pth'):
    # fix parameters of model
    args = {}
    args['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    args['metric_name'] = 'roc_auc'
    args['batch_size'] = 128
    args['num_epochs'] = 200
    args['d_model'] = 768
    args['n_layers'] = 6
    args['vocab_size'] = 47
    args['maxlen'] = 201
    args['d_k'] = 64
    args['d_v'] = 64
    args['d_ff'] = 768 * 4
    args['n_heads'] = 12
    args['global_labels_dim'] = 1
    args['atom_labels_dim'] = 15
    args['lr'] = 3e-5
    args['pretrain_layer'] = 6
    args['mode'] = 'higher'
    args['task_name'] = 'HIA'
    args['patience'] = 20
    args['times'] = 10
    args['pretrain_model'] = pretrain_model

    token_idx, global_label_list, atom_labels_list, atom_mask_list = construct_input_from_smiles(smiles)

    model = BERT_atom_embedding_generator(d_model=args['d_model'], n_layers=args['n_layers'], vocab_size=args['vocab_size'],
                                          maxlen=args['maxlen'], d_k=args['d_k'], d_v=args['d_v'], n_heads=args['n_heads'], d_ff=args['d_ff'],
                                          global_label_dim=args['global_labels_dim'], atom_label_dim=args['atom_labels_dim'], use_atom=False)
    stopper = EarlyStopping(pretrained_model=args['pretrain_model'],
                            pretrain_layer=args['pretrain_layer'],
                            mode=args['mode'])
    model.to(args['device'])
    stopper.load_pretrained_model(model)

    token_idx = torch.tensor([token_idx]).long().to(args['device'])
    atom_mask = atom_mask_list
    atom_mask_np = np.array(atom_mask)
    atom_mask_index = np.where(atom_mask_np == 1)
    h_global, h_atom = model(token_idx, atom_mask_index)
    h_global = h_global.cpu().squeeze().detach().numpy()
    h_atom = h_atom.cpu().squeeze().detach().numpy()
    return h_global, h_atom













