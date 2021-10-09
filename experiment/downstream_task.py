from experiment import build_data
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from experiment.my_nn import collate_data, EarlyStopping, run_a_train_global_epoch, run_an_eval_global_epoch,\
    set_random_seed, K_BERT_WCL, pos_weight
import os
import numpy as np
import pandas as pd
set_random_seed()


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
args['pretrain_layer'] = 5
args['mode'] = 'higher'
args['patience'] = 20
args['times'] = 10
args['pretrain_model'] = 'pretrain_k_bert_wcl_epoch_7.pth'
# args['pretrain_model'] = 'pretrain_k_bert_epoch_7.pth'

args['task_name_list'] = ['Pgp-sub', 'HIA', 'F(20%)', 'F(30%)',  'FDAMDD', 'CYP1A2-sub', 'CYP2C19-sub', 'CYP2C9-sub',
                          'CYP2D6-sub', 'CYP3A4-sub', 'T12', 'DILI', 'SkinSen', 'Carcinogenicity', 'Respiratory']

for task in args['task_name_list']:
    args['task_name'] = task
    args['data_path'] = '../data/task_data/' + args['task_name'] + '.npy'

    all_times_train_result = []
    all_times_val_result = []
    all_times_test_result = []
    result_pd = pd.DataFrame()
    result_pd['index'] = ['roc_auc', 'accuracy', 'sensitivity', 'specificity', 'f1-score', 'precision', 'recall',
                          'error rate', 'mcc']

    for time_id in range(args['times']):
        set_random_seed(2020+time_id)
        train_set, val_set, test_set, task_number = build_data.load_data_for_random_splited(
            data_path=args['data_path'], shuffle=True
        )
        print("Molecule graph is loaded!")
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=args['batch_size'],
                                  shuffle=True,
                                  collate_fn=collate_data)

        val_loader = DataLoader(dataset=val_set,
                                batch_size=args['batch_size'],
                                collate_fn=collate_data)

        test_loader = DataLoader(dataset=test_set,
                                 batch_size=args['batch_size'],
                                 collate_fn=collate_data)
        pos_weight_task = pos_weight(train_set)
        one_time_train_result = []
        one_time_val_result = []
        one_time_test_result = []
        print('***************************************************************************************************')
        print('{}, {}/{} time'.format(args['task_name'], time_id+1, args['times']))
        print('***************************************************************************************************')

        loss_criterion = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight_task.to(args['device']))
        model = K_BERT_WCL(d_model=args['d_model'], n_layers=args['n_layers'], vocab_size=args['vocab_size'],
                            maxlen=args['maxlen'], d_k=args['d_k'], d_v=args['d_v'], n_heads=args['n_heads'], d_ff=args['d_ff'],
                            global_label_dim=args['global_labels_dim'], atom_label_dim=args['atom_labels_dim'])
        stopper = EarlyStopping(patience=args['patience'], pretrained_model=args['pretrain_model'],
                                pretrain_layer=args['pretrain_layer'],
                                task_name=args['task_name']+'_downstream_k_bert_wcl', mode=args['mode'])
        model.to(args['device'])
        stopper.load_pretrained_model(model)
        optimizer = Adam(model.parameters(), lr=args['lr'])
        for epoch in range(args['num_epochs']):
            train_score = run_a_train_global_epoch(args, epoch, model, train_loader, loss_criterion, optimizer)
            # Validation and early stop
            _ = run_an_eval_global_epoch(args, model, train_loader)[0]
            val_score = run_an_eval_global_epoch(args, model, val_loader)[0]
            test_score = run_an_eval_global_epoch(args, model, test_loader)[0]
            if epoch < 5:
                early_stop = stopper.step(0, model)
            else:
                early_stop = stopper.step(val_score, model)
            print('epoch {:d}/{:d}, {}, lr: {:.6f},  train: {:.4f}, valid: {:.4f}, best valid {:.4f}, '
                  'test: {:.4f}'.format(
                  epoch + 1, args['num_epochs'], args['metric_name'], optimizer.param_groups[0]['lr'], train_score, val_score,
                  stopper.best_score, test_score))
            if early_stop:
                break
        stopper.load_checkpoint(model)
        train_score = run_an_eval_global_epoch(args, model, train_loader)[0]
        val_score = run_an_eval_global_epoch(args, model, val_loader)[0]
        test_score = run_an_eval_global_epoch(args, model, test_loader)[0]
        pred_name = 'prediction_' + str(time_id + 1)
        stop_test_list = run_an_eval_global_epoch(args, model, test_loader)
        stop_train_list = run_an_eval_global_epoch(args, model, train_loader)
        stop_val_list = run_an_eval_global_epoch(args, model, val_loader)
        result_pd['train_' + str(time_id + 1)] = stop_train_list
        result_pd['val_' + str(time_id + 1)] = stop_val_list
        result_pd['test_' + str(time_id + 1)] = stop_test_list
        print(result_pd[['index', 'train_' + str(time_id + 1), 'val_' + str(time_id + 1), 'test_' + str(time_id + 1)]])
        print('********************************{}, {}_times_result*******************************'.format(args['task_name'],
                                                                                                          time_id + 1))
        print("training_result:", round(train_score, 4))
        print("val_result:", round(val_score, 4))
        print("test_result:", round(test_score, 4))

        one_time_train_result.append(train_score)
        one_time_val_result.append(val_score)
        one_time_test_result.append(test_score)
        # except:
        #     task_number = task_number - 1
        all_times_train_result.append(round(np.array(one_time_train_result).mean(), 4))
        all_times_val_result.append(round(np.array(one_time_val_result).mean(), 4))
        all_times_test_result.append(round(np.array(one_time_test_result).mean(), 4))
        # except:
        #     print('{} times is failed!'.format(time_id+1))
        print("************************************{}_times_result************************************".format(
            time_id + 1))
        print('the train result of all tasks ({}): '.format(args['metric_name']), np.array(all_times_train_result))
        print('the average train result of all tasks ({}): {:.3f}'.format(args['metric_name'],
                                                                          np.array(all_times_train_result).mean()))
        print('the train result of all tasks (std): {:.3f}'.format(np.array(all_times_train_result).std()))
        print('the train result of all tasks (var): {:.3f}'.format(np.array(all_times_train_result).var()))

        print('the val result of all tasks ({}): '.format(args['metric_name']), np.array(all_times_val_result))
        print('the average val result of all tasks ({}): {:.3f}'.format(args['metric_name'],
                                                                        np.array(all_times_val_result).mean()))
        print('the val result of all tasks (std): {:.3f}'.format(np.array(all_times_val_result).std()))
        print('the val result of all tasks (var): {:.3f}'.format(np.array(all_times_val_result).var()))

        print('the test result of all tasks ({}):'.format(args['metric_name']), np.array(all_times_test_result))
        print('the average test result of all tasks ({}): {:.3f}'.format(args['metric_name'],
                                                                         np.array(all_times_test_result).mean()))
        print('the test result of all tasks (std): {:.3f}'.format(np.array(all_times_test_result).std()))
        print('the test result of all tasks (var): {:.3f}'.format(np.array(all_times_test_result).var()))
    result_pd.to_csv('../result/maccs/' + args['task_name'] + '_K_BERT_WCL_result.csv', index=False)



















