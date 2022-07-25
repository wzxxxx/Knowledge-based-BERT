from sklearn.metrics import roc_auc_score, mean_squared_error, precision_recall_curve, auc, r2_score
from sklearn import metrics
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import random
from torch import nn



def get_attn_pad_mask(seq_q):
    batch_size, seq_len = seq_q.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, maxlen):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(maxlen, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x).to('cuda')
        embedding = self.tok_embed(x) + self.pos_embed(pos)
        return self.norm(embedding)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        self.d_k = d_k
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        super(MultiHeadAttention, self).__init__()
        self.linear = nn.Linear(self.n_heads * self.d_v, self.d_model)
        self.layernorm = nn.LayerNorm(self.d_model)
        self.W_Q = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_heads)
    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        context = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.linear(context)
        return self.layernorm(output + residual)


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.layernorm = nn.LayerNorm(d_model)
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return self.layernorm.cuda()(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs


class K_BERT_WCL(nn.Module):
    def __init__(self, d_model, n_layers, vocab_size, maxlen, d_k, d_v, n_heads, d_ff, global_label_dim, atom_label_dim,
                 use_atom=False):
        super(K_BERT_WCL, self).__init__()
        self.maxlen = maxlen
        self.d_model = d_model
        self.use_atom = use_atom
        self.embedding = Embedding(vocab_size, self.d_model, maxlen)
        self.layers = nn.ModuleList([EncoderLayer(self.d_model, d_k, d_v, n_heads, d_ff) for _ in range(n_layers)])
        if self.use_atom:
            self.fc = nn.Sequential(
                nn.Dropout(0.),
                nn.Linear(self.d_model + self.d_model, self.d_model),
                nn.ReLU(),
                nn.BatchNorm1d(self.d_model))
            self.fc_weight = nn.Sequential(
                nn.Linear(self.d_model, 1),
                nn.Sigmoid())
        else:
            self.fc = nn.Sequential(
                nn.Dropout(0.),
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU(),
                nn.BatchNorm1d(self.d_model))
        self.classifier_global = nn.Linear(self.d_model, global_label_dim)
        self.classifier_atom = nn.Linear(self.d_model, atom_label_dim)

    def forward(self, input_ids):
        output = self.embedding(input_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids)
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)
        h_global = output[:, 0]
        if self.use_atom:
            h_atom = output[:, 1:]
            h_atom_weight = self.fc_weight(h_atom)
            h_atom_weight_expand = h_atom_weight.expand(h_atom.size())
            h_atom_mean = (h_atom*h_atom_weight_expand).mean(dim=1)
            h_mol = torch.cat([h_global, h_atom_mean], dim=1)
        else:
            h_mol = h_global
        h_embedding = self.fc(h_mol)
        logits_global = self.classifier_global(h_embedding)
        return logits_global

def cos_similar(p, q):
    sim_matrix = p.matmul(q.transpose(-2, -1))
    a = torch.norm(p, p=2, dim=-1)
    b = torch.norm(q, p=2, dim=-1)
    sim_matrix /= a.unsqueeze(-1)
    sim_matrix /= b.unsqueeze(-2)
    return sim_matrix


class K_BERT(nn.Module):
    def __init__(self, d_model, n_layers, vocab_size, maxlen, d_k, d_v, n_heads, d_ff, global_label_dim, atom_label_dim):
        super(K_BERT, self).__init__()
        self.maxlen = maxlen
        self.d_model = d_model
        self.embedding = Embedding(vocab_size, self.d_model, maxlen)
        self.layers = nn.ModuleList([EncoderLayer(self.d_model, d_k, d_v, n_heads, d_ff) for _ in range(n_layers)])
        self.fc_global = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Dropout(0.5),
            nn.Tanh(),
        )
        self.fc_atom = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Dropout(0.5),
            nn.Tanh(),
        )
        self.classifier_global = nn.Linear(self.d_model, global_label_dim)
        self.classifier_atom = nn.Linear(self.d_model, atom_label_dim)

    def forward(self, canonical_input_ids, aug_input_ids_1, aug_input_ids_2, aug_input_ids_3, aug_input_ids_4):
        canonical_output = self.embedding(canonical_input_ids)
        aug_output_1 = self.embedding(aug_input_ids_1)
        aug_output_2 = self.embedding(aug_input_ids_2)
        aug_output_3 = self.embedding(aug_input_ids_3)
        aug_output_4 = self.embedding(aug_input_ids_4)

        canonical_enc_self_attn_mask = get_attn_pad_mask(canonical_input_ids)
        aug_enc_self_attn_mask_1 = get_attn_pad_mask(aug_input_ids_1)
        aug_enc_self_attn_mask_2 = get_attn_pad_mask(aug_input_ids_2)
        aug_enc_self_attn_mask_3 = get_attn_pad_mask(aug_input_ids_3)
        aug_enc_self_attn_mask_4 = get_attn_pad_mask(aug_input_ids_4)

        for layer in self.layers:
            canonical_output = layer(canonical_output, canonical_enc_self_attn_mask)
            aug_output_1 = layer(aug_output_1, aug_enc_self_attn_mask_1)
            aug_output_2 = layer(aug_output_2, aug_enc_self_attn_mask_2)
            aug_output_3 = layer(aug_output_3, aug_enc_self_attn_mask_3)
            aug_output_4 = layer(aug_output_4, aug_enc_self_attn_mask_4)

        h_canonical_global = self.fc_global(canonical_output[:, 0])
        h_aug_global_1 = self.fc_global(aug_output_1[:, 0])
        h_aug_global_2 = self.fc_global(aug_output_2[:, 0])
        h_aug_global_3 = self.fc_global(aug_output_3[:, 0])
        h_aug_global_4 = self.fc_global(aug_output_4[:, 0])

        h_cos_1 = torch.cosine_similarity(canonical_output[:, 0], aug_output_1[:, 0], dim=1)
        h_cos_2 = torch.cosine_similarity(canonical_output[:, 0], aug_output_2[:, 0], dim=1)
        h_cos_3 = torch.cosine_similarity(canonical_output[:, 0], aug_output_3[:, 0], dim=1)
        h_cos_4 = torch.cosine_similarity(canonical_output[:, 0], aug_output_4[:, 0], dim=1)
        consensus_score = (torch.ones_like(h_cos_1)*4-h_cos_1 - h_cos_2 - h_cos_3 - h_cos_4)/8
        logits_canonical_global = self.classifier_global(h_canonical_global)
        logits_global_aug_1 = self.classifier_global(h_aug_global_1)
        logits_global_aug_2 = self.classifier_global(h_aug_global_2)
        logits_global_aug_3 = self.classifier_global(h_aug_global_3)
        logits_global_aug_4 = self.classifier_global(h_aug_global_4)
        canonical_cos_score_matric = torch.abs(cos_similar(canonical_output[:, 0], canonical_output[:, 0]))
        diagonal_cos_score_matric = torch.eye(canonical_cos_score_matric.size(0)).float().cuda()
        different_score = canonical_cos_score_matric - diagonal_cos_score_matric
        logits_global = torch.cat((logits_canonical_global, logits_global_aug_1, logits_global_aug_2,
                                   logits_global_aug_3, logits_global_aug_4), 1)

        h_atom = self.fc_atom(canonical_output[:, 1:])
        h_atom_emb = h_atom.reshape([len(canonical_output)*(self.maxlen - 1), self.d_model])
        logits_atom = self.classifier_atom(h_atom_emb)
        return logits_global, logits_atom, consensus_score, different_score


class BERT_atom_embedding_generator(nn.Module):
    def __init__(self, d_model, n_layers, vocab_size, maxlen, d_k, d_v, n_heads, d_ff, global_label_dim, atom_label_dim,
                 use_atom=False):
        super(BERT_atom_embedding_generator, self).__init__()
        self.maxlen = maxlen
        self.d_model = d_model
        self.use_atom = use_atom
        self.embedding = Embedding(vocab_size, self.d_model, maxlen)
        self.layers = nn.ModuleList([EncoderLayer(self.d_model, d_k, d_v, n_heads, d_ff) for _ in range(n_layers)])
        if self.use_atom:
            self.fc = nn.Sequential(
                nn.Dropout(0.),
                nn.Linear(self.d_model + self.d_model, self.d_model),
                nn.ReLU(),
                nn.BatchNorm1d(self.d_model))
            self.fc_weight = nn.Sequential(
                nn.Linear(self.d_model, 1),
                nn.Sigmoid())
        else:
            self.fc = nn.Sequential(
                nn.Dropout(0.),
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU(),
                nn.BatchNorm1d(self.d_model))
        self.classifier_global = nn.Linear(self.d_model, global_label_dim)
        self.classifier_atom = nn.Linear(self.d_model, atom_label_dim)

    def forward(self, input_ids, atom_mask):
        output = self.embedding(input_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids)
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)
        h_global = output[:, 0]
        h_atom = output[:, atom_mask]
        return h_global, h_atom


def set_random_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class Meter(object):
    """Track and summarize model performance on a dataset for
    (multi-label) binary classification."""
    def __init__(self):
        self.mask = []
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true, mask):
        """Update for the result of an iteration
        Parameters
        ----------
        y_pred : float32 tensor
            Predicted molecule labels with shape (B, T),
            B for batch size and T for the number of tasks
        y_true : float32 tensor
            Ground truth molecule labels with shape (B, T)
        mask : float32 tensor
            Mask for indicating the existence of ground
            truth labels with shape (B, T)
        """
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        if mask is None:
            mask.append(torch.ones_like(y_pred.detach().cpu()))
        else:
            self.mask.append(mask.detach().cpu())

    def roc_auc_score(self):
        """Compute roc-auc score for each task.
        Returns
        -------
        list of float
            roc-auc score for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(round(roc_auc_score(task_y_true, task_y_pred), 4))
        return scores

    def return_pred_true(self):
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        return y_pred, y_true

    def l1_loss(self, reduction):
        """Compute l1 loss for each task.
        Returns
        -------
        list of float
            l1 loss for all tasks
        reduction : str
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(F.l1_loss(task_y_true, task_y_pred, reduction=reduction).item())
        return scores

    def rmse(self):
        """Compute RMSE for each task.
        Returns
        -------
        list of float
            rmse for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(np.sqrt(F.mse_loss(task_y_pred, task_y_true).cpu().item()))
        return scores

    def mae(self):
        """Compute MAE for each task.
        Returns
        -------
        list of float
            mae for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(mean_squared_error(task_y_true, task_y_pred))
        return scores

    def r2(self):
        """Compute R2 for each task.
        Returns
        -------
        list of float
            r2 for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(round(r2_score(task_y_true, task_y_pred), 4))
        return scores

    def roc_precision_recall_score(self):
        """Compute AUC_PRC for each task.
        Returns
        -------
        list of float
            AUC_PRC for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            precision, recall, _thresholds = precision_recall_curve(task_y_true, task_y_pred)
            scores.append(auc(recall, precision))
        return scores

    def compute_metric(self, metric_name, reduction='mean'):
        """Compute metric for each task.
        Parameters
        ----------
        metric_name : str
            Name for the metric to compute.
        reduction : str
            Only comes into effect when the metric_name is l1_loss.
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        Returns
        -------
        list of float
            Metric value for each task
        """
        assert metric_name in ['roc_auc', 'l1', 'rmse', 'mae', 'roc_prc', 'r2', 'return_pred_true'], \
            'Expect metric name to be "roc_auc", "l1" or "rmse", "mae", "roc_prc", "r2", "return_pred_true", got {}'.format(metric_name)
        assert reduction in ['mean', 'sum']
        if metric_name == 'roc_auc':
            return self.roc_auc_score()
        if metric_name == 'l1':
            return self.l1_loss(reduction)
        if metric_name == 'rmse':
            return self.rmse()
        if metric_name == 'mae':
            return self.mae()
        if metric_name == 'roc_prc':
            return self.roc_precision_recall_score()
        if metric_name == 'r2':
            return self.r2()
        if metric_name == 'return_pred_true':
            return self.return_pred_true()


def collate_pretrain_data(data):
    tokens_idx, global_label_list, atom_labels_list, atom_mask_list = map(list, zip(*data))
    tokens_idx = torch.tensor(tokens_idx)
    global_label = torch.tensor(global_label_list)
    atom_labels = torch.tensor(atom_labels_list)
    atom_mask = torch.tensor(atom_mask_list)
    return tokens_idx, global_label, atom_labels, atom_mask


def collate_data(data):
    smiles, token_idx, global_label_list, mask = map(list, zip(*data))
    tokens_idx = torch.tensor(token_idx)
    global_label = torch.tensor(global_label_list)
    mask = torch.tensor(mask)
    return smiles, tokens_idx, global_label, mask


def pos_weight(train_set):
    smiles, tokens_idx, labels, mask = map(list, zip(*train_set))
    task_pos_weight_list = []
    for j in range(len(labels[1])):
        num_pos = 0
        num_impos = 0
        for i in labels:
            if i[j] == 1:
                num_pos = num_pos + 1
            if i[j] == 0:
                num_impos = num_impos + 1
        task_pos_weight = num_impos / (num_pos+0.00000001)
        task_pos_weight_list.append(task_pos_weight)
    return torch.tensor(task_pos_weight_list)


def run_a_pretrain_epoch(args, epoch, model, data_loader,
                         loss_criterion_global, loss_criterion_atom, optimizer):
    model.train()
    total_loss = 0
    for batch_id, batch_data in enumerate(data_loader):
        token_idx, global_labels, atom_labels, atom_mask = batch_data
        token_idx = token_idx.long().to(args['device'])
        global_labels = global_labels.float().to(args['device'])
        atom_labels = atom_labels[:, 1:].float().to(args['device'])
        atom_mask = atom_mask[:, 1:].float().to(args['device'])
        atom_labels = atom_labels.reshape([len(token_idx)*(args['maxlen']-1), args['atom_labels_dim']])
        atom_mask = atom_mask.reshape(len(token_idx)*(args['maxlen']-1), 1)
        logits_global, logits_atom = model(token_idx)
        loss = (loss_criterion_global(logits_global, global_labels).float()).mean() \
               + (loss_criterion_atom(logits_atom, atom_labels)*(atom_mask != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss*len(token_idx)
        print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
            epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss))
        del token_idx, global_labels, atom_labels, atom_mask, loss, logits_global, logits_atom
        torch.cuda.empty_cache()
    print('epoch {:d}/{:d}, pre-train loss {:.4f}'.format(
        epoch + 1, args['num_epochs'], total_loss))
    return total_loss


def run_a_contrastive_pretrain_epoch(args, epoch, model, data_loader,
                                   loss_criterion_global, loss_criterion_atom, optimizer):
    model.train()
    total_loss = 0
    for batch_id, batch_data in enumerate(data_loader):
        token_idx, global_labels, atom_labels, atom_mask = batch_data
        canonicaL_token_idx = token_idx[:, 0].long().to(args['device'])
        aug_token_idx_1 = token_idx[:, 1].long().to(args['device'])
        aug_token_idx_2 = token_idx[:, 2].long().to(args['device'])
        aug_token_idx_3 = token_idx[:, 3].long().to(args['device'])
        aug_token_idx_4 = token_idx[:, 4].long().to(args['device'])

        global_labels = global_labels.float().to(args['device'])
        global_labels = torch.cat((global_labels, global_labels, global_labels, global_labels, global_labels), 1)

        atom_labels = atom_labels[:, 1:].float().to(args['device'])
        atom_mask = atom_mask[:, 1:].float().to(args['device'])

        atom_labels = atom_labels.reshape([len(token_idx)*(args['maxlen']-1), args['atom_labels_dim']])
        atom_mask = atom_mask.reshape(len(token_idx)*(args['maxlen']-1), 1)

        logits_global, logits_atom, consensus_score, different_score = model(canonicaL_token_idx, aug_token_idx_1, aug_token_idx_2,
                                                            aug_token_idx_3, aug_token_idx_4)
        loss = (loss_criterion_global(logits_global, global_labels).float()).mean() \
                + (loss_criterion_atom(logits_atom, atom_labels)*(atom_mask != 0).float()).mean()\
                + consensus_score.mean() + different_score.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss*len(token_idx)
        print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}, consensus_loss {:.4f}, different_loss {:.4f}, global_loss {:.4f}, atom_loss {:.4f}'.format(
            epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss, consensus_score.mean(), different_score.mean(),
            (loss_criterion_global(logits_global, global_labels).float()).mean(),
            (loss_criterion_atom(logits_atom, atom_labels)*(atom_mask != 0).float()).mean()))
        del token_idx, global_labels, atom_labels, atom_mask, loss, logits_global, logits_atom
        torch.cuda.empty_cache()
    print('epoch {:d}/{:d}, pre-train loss {:.4f}'.format(
        epoch + 1, args['num_epochs'], total_loss))
    return total_loss


def run_a_contrastive_R_S_pretrain_epoch(args, epoch, model, data_loader,
                                         loss_criterion_global, loss_criterion_atom, optimizer):
    model.train()
    total_loss = 0
    for batch_id, batch_data in enumerate(data_loader):
        token_idx, global_labels, atom_labels, atom_mask = batch_data
        canonicaL_token_idx = token_idx[:, 0].long().to(args['device'])
        aug_token_idx_1 = token_idx[:, 1].long().to(args['device'])
        aug_token_idx_2 = token_idx[:, 2].long().to(args['device'])
        aug_token_idx_3 = token_idx[:, 3].long().to(args['device'])
        aug_token_idx_4 = token_idx[:, 4].long().to(args['device'])

        global_labels = global_labels.unsqueeze(1).float().to(args['device'])
        global_labels = torch.cat((global_labels, global_labels, global_labels, global_labels, global_labels), 1)

        atom_labels = atom_labels[:, 1:].float().to(args['device'])
        atom_mask = atom_mask[:, 1:].float().to(args['device'])

        atom_labels = atom_labels.reshape([len(token_idx)*(args['maxlen']-1), args['atom_labels_dim']])
        atom_mask = atom_mask.reshape(len(token_idx)*(args['maxlen']-1), 1)

        logits_global, logits_atom, consensus_score, different_score = model(canonicaL_token_idx, aug_token_idx_1, aug_token_idx_2,
                                                            aug_token_idx_3, aug_token_idx_4)
        loss = (loss_criterion_global(logits_global, global_labels).float()).mean() \
                + (loss_criterion_atom(logits_atom, atom_labels)*(atom_mask != 0).float()).mean()\
                + consensus_score.mean() + different_score.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss*len(token_idx)
        print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}, consensus_loss {:.4f}, different_loss {:.4f}, global_loss {:.4f}, atom_loss {:.4f}'.format(
            epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss, consensus_score.mean(), different_score.mean(),
            (loss_criterion_global(logits_global, global_labels).float()).mean(),
            (loss_criterion_atom(logits_atom, atom_labels)*(atom_mask != 0).float()).mean()))
        del token_idx, global_labels, atom_labels, atom_mask, loss, logits_global, logits_atom
        torch.cuda.empty_cache()
    print('epoch {:d}/{:d}, pre-train loss {:.4f}'.format(
        epoch + 1, args['num_epochs'], total_loss))
    return total_loss


def run_a_train_global_epoch(args, epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        smiles, token_idx, global_labels, mask = batch_data
        token_idx = token_idx.long().to(args['device'])
        global_labels = global_labels.float().to(args['device'])
        mask = mask.float().to(args['device'])
        logits_global = model(token_idx)
        loss = (loss_criterion(logits_global, global_labels).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
               epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
        train_meter.update(logits_global, global_labels, mask=mask)
        del token_idx, global_labels, loss
        torch.cuda.empty_cache()
    train_score = np.mean(train_meter.compute_metric(args['metric_name']))
    return train_score


def run_a_train_contrastive_global_epoch(args, epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        smiles, token_idx, global_labels, mask = batch_data
        canonicaL_token_idx = token_idx[:, 0].long().to(args['device'])
        aug_token_idx_1 = token_idx[:, 1].long().to(args['device'])
        aug_token_idx_2 = token_idx[:, 2].long().to(args['device'])
        aug_token_idx_3 = token_idx[:, 3].long().to(args['device'])
        aug_token_idx_4 = token_idx[:, 4].long().to(args['device'])

        global_labels = global_labels.float().to(args['device'])
        global_labels = torch.cat((global_labels, global_labels, global_labels, global_labels, global_labels), 1)
        mask = mask.float().to(args['device'])
        mask = torch.cat((mask, mask, mask, mask, mask), 1)
        logits_global, _, consensus_score, different_score = model(canonicaL_token_idx, aug_token_idx_1, aug_token_idx_2,
                                                            aug_token_idx_3, aug_token_idx_4)
        loss = (loss_criterion(logits_global, global_labels).float()).mean() + consensus_score.mean() + different_score.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}, consenesu_loss {:.4f}, different_loss {:.4f}'.format(
               epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item(), consensus_score.mean(),
            different_score.mean()))
        train_meter.update(logits_global, global_labels, mask=mask)
        del token_idx, global_labels, loss
        torch.cuda.empty_cache()
    train_score = np.mean(train_meter.compute_metric(args['metric_name']))
    return train_score


def sesp_score(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] == 1:
            tp = tp + 1
        if y_true[i] == y_pred[i] == 0:
            tn = tn + 1
        if y_true[i] == 0 and y_pred[i] == 1:
            fp = fp + 1
        if y_true[i] == 1 and y_pred[i] == 0:
            fn = fn + 1
    sensitivity = round(tp/(tp+fn), 4)
    specificity = round(tn/(tn+fp), 4)
    return sensitivity, specificity


def run_an_eval_global_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, token_idx, global_labels, mask = batch_data
            token_idx = token_idx.long().to(args['device'])
            mask = mask.float().to(args['device'])
            global_labels = global_labels.float().to(args['device'])
            logits_global = model(token_idx)
            eval_meter.update(logits_global, global_labels, mask=mask)
            del token_idx, global_labels
            torch.cuda.empty_cache()
    y_pred, y_true = eval_meter.compute_metric('return_pred_true')
    y_true_list = y_true.squeeze(dim=1).tolist()
    y_pred_list = torch.sigmoid(y_pred).squeeze(dim=1).tolist()
    # save prediction
    y_pred_label = [1 if x >= 0.5 else 0 for x in y_pred_list]
    auc = metrics.roc_auc_score(y_true_list, y_pred_list)
    accuracy = metrics.accuracy_score(y_true_list, y_pred_label)
    se, sp = sesp_score(y_true_list, y_pred_label)
    pre, rec, f1, sup = metrics.precision_recall_fscore_support(y_true_list, y_pred_label)
    mcc = metrics.matthews_corrcoef(y_true_list, y_pred_label)
    f1 = f1[1]
    rec = rec[1]
    pre = pre[1]
    err = 1 - accuracy
    result = [auc, accuracy, se, sp, f1, pre, rec, err, mcc]
    return result


def run_an_eval_AUG_global_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, token_idx, global_labels, mask = batch_data
            token_idx = token_idx.long().to(args['device'])
            mask = mask.float().to(args['device'])
            global_labels = global_labels.float().to(args['device'])
            logits_global = model(token_idx)
            if batch_id == 0:
                all_global_labels = global_labels
                all_logits_global = logits_global
                all_mask = mask
            else:
                all_global_labels = torch.cat((all_global_labels, global_labels), 0)
                all_logits_global = torch.cat((all_logits_global, logits_global), 0)
                all_mask = torch.cat((all_mask, mask), 0)
    mask_ensemble = all_mask.reshape([args['aug_times'], int(len(all_mask) / args['aug_times'])])
    global_labels_ensemble = all_global_labels.reshape([args['aug_times'], int(len(all_global_labels) / args['aug_times'])])
    logits_global_ensemble = all_logits_global.reshape([args['aug_times'], int(len(all_logits_global) / args['aug_times'])])
    final_labels = torch.mean(global_labels_ensemble, axis=0).unsqueeze(1)
    final_logits = torch.mean(logits_global_ensemble, axis=0).unsqueeze(1)
    final_mask = torch.mean(mask_ensemble, axis=0).unsqueeze(1)
    eval_meter.update(final_logits, final_labels, mask=final_mask)
    del token_idx, global_labels, global_labels_ensemble, logits_global_ensemble, all_global_labels, all_logits_global
    torch.cuda.empty_cache()
    y_pred, y_true = eval_meter.compute_metric('return_pred_true')
    y_true_list = y_true.squeeze(dim=1).tolist()
    y_pred_list = torch.sigmoid(y_pred).squeeze(dim=1).tolist()
    # save prediction
    y_pred_label = [1 if x >= 0.5 else 0 for x in y_pred_list]
    auc = metrics.roc_auc_score(y_true_list, y_pred_list)
    accuracy = metrics.accuracy_score(y_true_list, y_pred_label)
    se, sp = sesp_score(y_true_list, y_pred_label)
    pre, rec, f1, sup = metrics.precision_recall_fscore_support(y_true_list, y_pred_label)
    mcc = metrics.matthews_corrcoef(y_true_list, y_pred_label)
    f1 = f1[1]
    rec = rec[1]
    pre = pre[1]
    err = 1 - accuracy
    result = [auc, accuracy, se, sp, f1, pre, rec, err, mcc]
    return result


def run_an_eval_contrastive_global_epoch(args, model, data_loader, save_path=None):
    model.eval()
    eval_meter = Meter()
    smiles_list = []
    cos_similarity_list = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, token_idx, global_labels, mask = batch_data
            canonicaL_token_idx = token_idx[:, 0].long().to(args['device'])
            aug_token_idx_1 = token_idx[:, 1].long().to(args['device'])
            aug_token_idx_2 = token_idx[:, 2].long().to(args['device'])
            aug_token_idx_3 = token_idx[:, 3].long().to(args['device'])
            aug_token_idx_4 = token_idx[:, 4].long().to(args['device'])

            mask = mask.float().to(args['device'])
            global_labels = global_labels.float().to(args['device'])

            logits_global, _, consensus_score, _ = model(canonicaL_token_idx, aug_token_idx_1, aug_token_idx_2,
                                                            aug_token_idx_3, aug_token_idx_4)
            logits_global_ensemble = torch.mean(logits_global, 1).unsqueeze(1)
            eval_meter.update(logits_global_ensemble, global_labels, mask=mask)
            if save_path is not None:
                cos_similarity = (torch.ones_like(consensus_score) - consensus_score*2).cpu().numpy().tolist()
                smiles_list = smiles_list + smiles
                cos_similarity_list = cos_similarity_list + cos_similarity
            del token_idx, global_labels
            torch.cuda.empty_cache()
    y_pred, y_true = eval_meter.compute_metric('return_pred_true')
    y_true_list = y_true.squeeze(dim=1).tolist()
    y_pred_list = torch.sigmoid(y_pred).squeeze(dim=1).tolist()
    if save_path is not None:
        prediction = pd.DataFrame()
        prediction['smiles'] = smiles_list
        prediction['prediction'] = y_pred_list
        prediction['label'] = y_true_list
        prediction['cos_similarity'] = cos_similarity_list
        prediction.to_csv(save_path, index=False)
        print(prediction)
    # save prediction
    y_pred_label = [1 if x >= 0.5 else 0 for x in y_pred_list]
    auc = metrics.roc_auc_score(y_true_list, y_pred_list)
    accuracy = metrics.accuracy_score(y_true_list, y_pred_label)
    se, sp = sesp_score(y_true_list, y_pred_label)
    pre, rec, f1, sup = metrics.precision_recall_fscore_support(y_true_list, y_pred_label)
    mcc = metrics.matthews_corrcoef(y_true_list, y_pred_label)
    f1 = f1[1]
    rec = rec[1]
    pre = pre[1]
    err = 1 - accuracy
    result = [auc, accuracy, se, sp, f1, pre, rec, err, mcc]
    return result


class EarlyStopping(object):
    def __init__(self, pretrained_model='Null_early_stop.pth',
                 pretrain_layer=6, mode='higher', patience=10, task_name="None"):
        assert mode in ['higher', 'lower']
        self.pretrain_layer = pretrain_layer
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = '../model/{}_early_stop.pth'.format(task_name)
        self.pretrain_save_filename = '../model/pretrain_{}_epoch_'.format(task_name)
        self.best_score = None
        self.early_stop = False
        self.pretrained_model = '../model/{}'.format(pretrained_model)

    def _check_higher(self, score, prev_best_score):
        return (score > prev_best_score)

    def _check_lower(self, score, prev_best_score):
        return (score < prev_best_score)

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                'EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def pretrain_step(self, epoch, model):
        print('Pretrain epoch {} is finished and the model is saved'.format(epoch))
        self.pretrain_save_checkpoint(epoch, model)

    def pretrain_save_checkpoint(self, epoch, model):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model.state_dict()}, self.pretrain_save_filename + str(epoch) + '.pth')
        # print(self.filename)

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model.state_dict()}, self.filename)
        # print(self.filename)

    def load_checkpoint(self, model):
        '''Load model saved with early stopping.'''
        # model.load_state_dict(torch.load(self.filename)['model_state_dict'])
        model.load_state_dict(torch.load(self.filename, map_location=torch.device('cpu'))['model_state_dict'])

    def load_pretrained_model(self, model):
        if self.pretrain_layer == 1:
            pretrained_parameters = ['embedding.tok_embed.weight', 'embedding.pos_embed.weight', 'embedding.norm.weight', 'embedding.norm.bias', 'layers.0.enc_self_attn.linear.weight', 'layers.0.enc_self_attn.linear.bias', 'layers.0.enc_self_attn.layernorm.weight', 'layers.0.enc_self_attn.layernorm.bias', 'layers.0.enc_self_attn.W_Q.weight', 'layers.0.enc_self_attn.W_Q.bias', 'layers.0.enc_self_attn.W_K.weight', 'layers.0.enc_self_attn.W_K.bias', 'layers.0.enc_self_attn.W_V.weight', 'layers.0.enc_self_attn.W_V.bias', 'layers.0.pos_ffn.fc.0.weight', 'layers.0.pos_ffn.fc.2.weight', 'layers.0.pos_ffn.layernorm.weight', 'layers.0.pos_ffn.layernorm.bias']

        elif self.pretrain_layer == 2:
            pretrained_parameters = ['embedding.tok_embed.weight', 'embedding.pos_embed.weight', 'embedding.norm.weight', 'embedding.norm.bias', 'layers.0.enc_self_attn.linear.weight', 'layers.0.enc_self_attn.linear.bias', 'layers.0.enc_self_attn.layernorm.weight', 'layers.0.enc_self_attn.layernorm.bias', 'layers.0.enc_self_attn.W_Q.weight', 'layers.0.enc_self_attn.W_Q.bias', 'layers.0.enc_self_attn.W_K.weight', 'layers.0.enc_self_attn.W_K.bias', 'layers.0.enc_self_attn.W_V.weight', 'layers.0.enc_self_attn.W_V.bias', 'layers.0.pos_ffn.fc.0.weight', 'layers.0.pos_ffn.fc.2.weight', 'layers.0.pos_ffn.layernorm.weight', 'layers.0.pos_ffn.layernorm.bias', 'layers.1.enc_self_attn.linear.weight', 'layers.1.enc_self_attn.linear.bias', 'layers.1.enc_self_attn.layernorm.weight', 'layers.1.enc_self_attn.layernorm.bias', 'layers.1.enc_self_attn.W_Q.weight', 'layers.1.enc_self_attn.W_Q.bias', 'layers.1.enc_self_attn.W_K.weight', 'layers.1.enc_self_attn.W_K.bias', 'layers.1.enc_self_attn.W_V.weight', 'layers.1.enc_self_attn.W_V.bias', 'layers.1.pos_ffn.fc.0.weight', 'layers.1.pos_ffn.fc.2.weight', 'layers.1.pos_ffn.layernorm.weight', 'layers.1.pos_ffn.layernorm.bias']

        elif self.pretrain_layer == 3:
            pretrained_parameters = ['embedding.tok_embed.weight', 'embedding.pos_embed.weight', 'embedding.norm.weight', 'embedding.norm.bias', 'layers.0.enc_self_attn.linear.weight', 'layers.0.enc_self_attn.linear.bias', 'layers.0.enc_self_attn.layernorm.weight', 'layers.0.enc_self_attn.layernorm.bias', 'layers.0.enc_self_attn.W_Q.weight', 'layers.0.enc_self_attn.W_Q.bias', 'layers.0.enc_self_attn.W_K.weight', 'layers.0.enc_self_attn.W_K.bias', 'layers.0.enc_self_attn.W_V.weight', 'layers.0.enc_self_attn.W_V.bias', 'layers.0.pos_ffn.fc.0.weight', 'layers.0.pos_ffn.fc.2.weight', 'layers.0.pos_ffn.layernorm.weight', 'layers.0.pos_ffn.layernorm.bias', 'layers.1.enc_self_attn.linear.weight', 'layers.1.enc_self_attn.linear.bias', 'layers.1.enc_self_attn.layernorm.weight', 'layers.1.enc_self_attn.layernorm.bias', 'layers.1.enc_self_attn.W_Q.weight', 'layers.1.enc_self_attn.W_Q.bias', 'layers.1.enc_self_attn.W_K.weight', 'layers.1.enc_self_attn.W_K.bias', 'layers.1.enc_self_attn.W_V.weight', 'layers.1.enc_self_attn.W_V.bias', 'layers.1.pos_ffn.fc.0.weight', 'layers.1.pos_ffn.fc.2.weight', 'layers.1.pos_ffn.layernorm.weight', 'layers.1.pos_ffn.layernorm.bias', 'layers.2.enc_self_attn.linear.weight', 'layers.2.enc_self_attn.linear.bias', 'layers.2.enc_self_attn.layernorm.weight', 'layers.2.enc_self_attn.layernorm.bias', 'layers.2.enc_self_attn.W_Q.weight', 'layers.2.enc_self_attn.W_Q.bias', 'layers.2.enc_self_attn.W_K.weight', 'layers.2.enc_self_attn.W_K.bias', 'layers.2.enc_self_attn.W_V.weight', 'layers.2.enc_self_attn.W_V.bias', 'layers.2.pos_ffn.fc.0.weight', 'layers.2.pos_ffn.fc.2.weight', 'layers.2.pos_ffn.layernorm.weight', 'layers.2.pos_ffn.layernorm.bias']

        elif self.pretrain_layer == 4:
            pretrained_parameters = ['embedding.tok_embed.weight', 'embedding.pos_embed.weight', 'embedding.norm.weight', 'embedding.norm.bias', 'layers.0.enc_self_attn.linear.weight', 'layers.0.enc_self_attn.linear.bias', 'layers.0.enc_self_attn.layernorm.weight', 'layers.0.enc_self_attn.layernorm.bias', 'layers.0.enc_self_attn.W_Q.weight', 'layers.0.enc_self_attn.W_Q.bias', 'layers.0.enc_self_attn.W_K.weight', 'layers.0.enc_self_attn.W_K.bias', 'layers.0.enc_self_attn.W_V.weight', 'layers.0.enc_self_attn.W_V.bias', 'layers.0.pos_ffn.fc.0.weight', 'layers.0.pos_ffn.fc.2.weight', 'layers.0.pos_ffn.layernorm.weight', 'layers.0.pos_ffn.layernorm.bias', 'layers.1.enc_self_attn.linear.weight', 'layers.1.enc_self_attn.linear.bias', 'layers.1.enc_self_attn.layernorm.weight', 'layers.1.enc_self_attn.layernorm.bias', 'layers.1.enc_self_attn.W_Q.weight', 'layers.1.enc_self_attn.W_Q.bias', 'layers.1.enc_self_attn.W_K.weight', 'layers.1.enc_self_attn.W_K.bias', 'layers.1.enc_self_attn.W_V.weight', 'layers.1.enc_self_attn.W_V.bias', 'layers.1.pos_ffn.fc.0.weight', 'layers.1.pos_ffn.fc.2.weight', 'layers.1.pos_ffn.layernorm.weight', 'layers.1.pos_ffn.layernorm.bias', 'layers.2.enc_self_attn.linear.weight', 'layers.2.enc_self_attn.linear.bias', 'layers.2.enc_self_attn.layernorm.weight', 'layers.2.enc_self_attn.layernorm.bias', 'layers.2.enc_self_attn.W_Q.weight', 'layers.2.enc_self_attn.W_Q.bias', 'layers.2.enc_self_attn.W_K.weight', 'layers.2.enc_self_attn.W_K.bias', 'layers.2.enc_self_attn.W_V.weight', 'layers.2.enc_self_attn.W_V.bias', 'layers.2.pos_ffn.fc.0.weight', 'layers.2.pos_ffn.fc.2.weight', 'layers.2.pos_ffn.layernorm.weight', 'layers.2.pos_ffn.layernorm.bias', 'layers.3.enc_self_attn.linear.weight', 'layers.3.enc_self_attn.linear.bias', 'layers.3.enc_self_attn.layernorm.weight', 'layers.3.enc_self_attn.layernorm.bias', 'layers.3.enc_self_attn.W_Q.weight', 'layers.3.enc_self_attn.W_Q.bias', 'layers.3.enc_self_attn.W_K.weight', 'layers.3.enc_self_attn.W_K.bias', 'layers.3.enc_self_attn.W_V.weight', 'layers.3.enc_self_attn.W_V.bias', 'layers.3.pos_ffn.fc.0.weight', 'layers.3.pos_ffn.fc.2.weight', 'layers.3.pos_ffn.layernorm.weight', 'layers.3.pos_ffn.layernorm.bias']

        elif self.pretrain_layer == 5:
            pretrained_parameters = ['embedding.tok_embed.weight', 'embedding.pos_embed.weight', 'embedding.norm.weight', 'embedding.norm.bias', 'layers.0.enc_self_attn.linear.weight', 'layers.0.enc_self_attn.linear.bias', 'layers.0.enc_self_attn.layernorm.weight', 'layers.0.enc_self_attn.layernorm.bias', 'layers.0.enc_self_attn.W_Q.weight', 'layers.0.enc_self_attn.W_Q.bias', 'layers.0.enc_self_attn.W_K.weight', 'layers.0.enc_self_attn.W_K.bias', 'layers.0.enc_self_attn.W_V.weight', 'layers.0.enc_self_attn.W_V.bias', 'layers.0.pos_ffn.fc.0.weight', 'layers.0.pos_ffn.fc.2.weight', 'layers.0.pos_ffn.layernorm.weight', 'layers.0.pos_ffn.layernorm.bias', 'layers.1.enc_self_attn.linear.weight', 'layers.1.enc_self_attn.linear.bias', 'layers.1.enc_self_attn.layernorm.weight', 'layers.1.enc_self_attn.layernorm.bias', 'layers.1.enc_self_attn.W_Q.weight', 'layers.1.enc_self_attn.W_Q.bias', 'layers.1.enc_self_attn.W_K.weight', 'layers.1.enc_self_attn.W_K.bias', 'layers.1.enc_self_attn.W_V.weight', 'layers.1.enc_self_attn.W_V.bias', 'layers.1.pos_ffn.fc.0.weight', 'layers.1.pos_ffn.fc.2.weight', 'layers.1.pos_ffn.layernorm.weight', 'layers.1.pos_ffn.layernorm.bias', 'layers.2.enc_self_attn.linear.weight', 'layers.2.enc_self_attn.linear.bias', 'layers.2.enc_self_attn.layernorm.weight', 'layers.2.enc_self_attn.layernorm.bias', 'layers.2.enc_self_attn.W_Q.weight', 'layers.2.enc_self_attn.W_Q.bias', 'layers.2.enc_self_attn.W_K.weight', 'layers.2.enc_self_attn.W_K.bias', 'layers.2.enc_self_attn.W_V.weight', 'layers.2.enc_self_attn.W_V.bias', 'layers.2.pos_ffn.fc.0.weight', 'layers.2.pos_ffn.fc.2.weight', 'layers.2.pos_ffn.layernorm.weight', 'layers.2.pos_ffn.layernorm.bias', 'layers.3.enc_self_attn.linear.weight', 'layers.3.enc_self_attn.linear.bias', 'layers.3.enc_self_attn.layernorm.weight', 'layers.3.enc_self_attn.layernorm.bias', 'layers.3.enc_self_attn.W_Q.weight', 'layers.3.enc_self_attn.W_Q.bias', 'layers.3.enc_self_attn.W_K.weight', 'layers.3.enc_self_attn.W_K.bias', 'layers.3.enc_self_attn.W_V.weight', 'layers.3.enc_self_attn.W_V.bias', 'layers.3.pos_ffn.fc.0.weight', 'layers.3.pos_ffn.fc.2.weight', 'layers.3.pos_ffn.layernorm.weight', 'layers.3.pos_ffn.layernorm.bias', 'layers.4.enc_self_attn.linear.weight', 'layers.4.enc_self_attn.linear.bias', 'layers.4.enc_self_attn.layernorm.weight', 'layers.4.enc_self_attn.layernorm.bias', 'layers.4.enc_self_attn.W_Q.weight', 'layers.4.enc_self_attn.W_Q.bias', 'layers.4.enc_self_attn.W_K.weight', 'layers.4.enc_self_attn.W_K.bias', 'layers.4.enc_self_attn.W_V.weight', 'layers.4.enc_self_attn.W_V.bias', 'layers.4.pos_ffn.fc.0.weight', 'layers.4.pos_ffn.fc.2.weight', 'layers.4.pos_ffn.layernorm.weight', 'layers.4.pos_ffn.layernorm.bias']
        elif self.pretrain_layer == 6:
            pretrained_parameters = ['embedding.tok_embed.weight', 'embedding.pos_embed.weight', 'embedding.norm.weight', 'embedding.norm.bias', 'layers.0.enc_self_attn.linear.weight', 'layers.0.enc_self_attn.linear.bias', 'layers.0.enc_self_attn.layernorm.weight', 'layers.0.enc_self_attn.layernorm.bias', 'layers.0.enc_self_attn.W_Q.weight', 'layers.0.enc_self_attn.W_Q.bias', 'layers.0.enc_self_attn.W_K.weight', 'layers.0.enc_self_attn.W_K.bias', 'layers.0.enc_self_attn.W_V.weight', 'layers.0.enc_self_attn.W_V.bias', 'layers.0.pos_ffn.fc.0.weight', 'layers.0.pos_ffn.fc.2.weight', 'layers.0.pos_ffn.layernorm.weight', 'layers.0.pos_ffn.layernorm.bias', 'layers.1.enc_self_attn.linear.weight', 'layers.1.enc_self_attn.linear.bias', 'layers.1.enc_self_attn.layernorm.weight', 'layers.1.enc_self_attn.layernorm.bias', 'layers.1.enc_self_attn.W_Q.weight', 'layers.1.enc_self_attn.W_Q.bias', 'layers.1.enc_self_attn.W_K.weight', 'layers.1.enc_self_attn.W_K.bias', 'layers.1.enc_self_attn.W_V.weight', 'layers.1.enc_self_attn.W_V.bias', 'layers.1.pos_ffn.fc.0.weight', 'layers.1.pos_ffn.fc.2.weight', 'layers.1.pos_ffn.layernorm.weight', 'layers.1.pos_ffn.layernorm.bias', 'layers.2.enc_self_attn.linear.weight', 'layers.2.enc_self_attn.linear.bias', 'layers.2.enc_self_attn.layernorm.weight', 'layers.2.enc_self_attn.layernorm.bias', 'layers.2.enc_self_attn.W_Q.weight', 'layers.2.enc_self_attn.W_Q.bias', 'layers.2.enc_self_attn.W_K.weight', 'layers.2.enc_self_attn.W_K.bias', 'layers.2.enc_self_attn.W_V.weight', 'layers.2.enc_self_attn.W_V.bias', 'layers.2.pos_ffn.fc.0.weight', 'layers.2.pos_ffn.fc.2.weight', 'layers.2.pos_ffn.layernorm.weight', 'layers.2.pos_ffn.layernorm.bias', 'layers.3.enc_self_attn.linear.weight', 'layers.3.enc_self_attn.linear.bias', 'layers.3.enc_self_attn.layernorm.weight', 'layers.3.enc_self_attn.layernorm.bias', 'layers.3.enc_self_attn.W_Q.weight', 'layers.3.enc_self_attn.W_Q.bias', 'layers.3.enc_self_attn.W_K.weight', 'layers.3.enc_self_attn.W_K.bias', 'layers.3.enc_self_attn.W_V.weight', 'layers.3.enc_self_attn.W_V.bias', 'layers.3.pos_ffn.fc.0.weight', 'layers.3.pos_ffn.fc.2.weight', 'layers.3.pos_ffn.layernorm.weight', 'layers.3.pos_ffn.layernorm.bias', 'layers.4.enc_self_attn.linear.weight', 'layers.4.enc_self_attn.linear.bias', 'layers.4.enc_self_attn.layernorm.weight', 'layers.4.enc_self_attn.layernorm.bias', 'layers.4.enc_self_attn.W_Q.weight', 'layers.4.enc_self_attn.W_Q.bias', 'layers.4.enc_self_attn.W_K.weight', 'layers.4.enc_self_attn.W_K.bias', 'layers.4.enc_self_attn.W_V.weight', 'layers.4.enc_self_attn.W_V.bias', 'layers.4.pos_ffn.fc.0.weight', 'layers.4.pos_ffn.fc.2.weight', 'layers.4.pos_ffn.layernorm.weight', 'layers.4.pos_ffn.layernorm.bias', 'layers.5.enc_self_attn.linear.weight', 'layers.5.enc_self_attn.linear.bias', 'layers.5.enc_self_attn.layernorm.weight', 'layers.5.enc_self_attn.layernorm.bias', 'layers.5.enc_self_attn.W_Q.weight', 'layers.5.enc_self_attn.W_Q.bias', 'layers.5.enc_self_attn.W_K.weight', 'layers.5.enc_self_attn.W_K.bias', 'layers.5.enc_self_attn.W_V.weight', 'layers.5.enc_self_attn.W_V.bias', 'layers.5.pos_ffn.fc.0.weight', 'layers.5.pos_ffn.fc.2.weight', 'layers.5.pos_ffn.layernorm.weight', 'layers.5.pos_ffn.layernorm.bias']
        elif self.pretrain_layer == 7:
            pretrained_parameters = ['embedding.tok_embed.weight', 'embedding.pos_embed.weight', 'embedding.norm.weight', 'embedding.norm.bias', 'layers.0.enc_self_attn.linear.weight', 'layers.0.enc_self_attn.linear.bias', 'layers.0.enc_self_attn.layernorm.weight', 'layers.0.enc_self_attn.layernorm.bias', 'layers.0.enc_self_attn.W_Q.weight', 'layers.0.enc_self_attn.W_Q.bias', 'layers.0.enc_self_attn.W_K.weight', 'layers.0.enc_self_attn.W_K.bias', 'layers.0.enc_self_attn.W_V.weight', 'layers.0.enc_self_attn.W_V.bias', 'layers.0.pos_ffn.fc.0.weight', 'layers.0.pos_ffn.fc.2.weight', 'layers.0.pos_ffn.layernorm.weight', 'layers.0.pos_ffn.layernorm.bias', 'layers.1.enc_self_attn.linear.weight', 'layers.1.enc_self_attn.linear.bias', 'layers.1.enc_self_attn.layernorm.weight', 'layers.1.enc_self_attn.layernorm.bias', 'layers.1.enc_self_attn.W_Q.weight', 'layers.1.enc_self_attn.W_Q.bias', 'layers.1.enc_self_attn.W_K.weight', 'layers.1.enc_self_attn.W_K.bias', 'layers.1.enc_self_attn.W_V.weight', 'layers.1.enc_self_attn.W_V.bias', 'layers.1.pos_ffn.fc.0.weight', 'layers.1.pos_ffn.fc.2.weight', 'layers.1.pos_ffn.layernorm.weight', 'layers.1.pos_ffn.layernorm.bias', 'layers.2.enc_self_attn.linear.weight', 'layers.2.enc_self_attn.linear.bias', 'layers.2.enc_self_attn.layernorm.weight', 'layers.2.enc_self_attn.layernorm.bias', 'layers.2.enc_self_attn.W_Q.weight', 'layers.2.enc_self_attn.W_Q.bias', 'layers.2.enc_self_attn.W_K.weight', 'layers.2.enc_self_attn.W_K.bias', 'layers.2.enc_self_attn.W_V.weight', 'layers.2.enc_self_attn.W_V.bias', 'layers.2.pos_ffn.fc.0.weight', 'layers.2.pos_ffn.fc.2.weight', 'layers.2.pos_ffn.layernorm.weight', 'layers.2.pos_ffn.layernorm.bias', 'layers.3.enc_self_attn.linear.weight', 'layers.3.enc_self_attn.linear.bias', 'layers.3.enc_self_attn.layernorm.weight', 'layers.3.enc_self_attn.layernorm.bias', 'layers.3.enc_self_attn.W_Q.weight', 'layers.3.enc_self_attn.W_Q.bias', 'layers.3.enc_self_attn.W_K.weight', 'layers.3.enc_self_attn.W_K.bias', 'layers.3.enc_self_attn.W_V.weight', 'layers.3.enc_self_attn.W_V.bias', 'layers.3.pos_ffn.fc.0.weight', 'layers.3.pos_ffn.fc.2.weight', 'layers.3.pos_ffn.layernorm.weight', 'layers.3.pos_ffn.layernorm.bias', 'layers.4.enc_self_attn.linear.weight', 'layers.4.enc_self_attn.linear.bias', 'layers.4.enc_self_attn.layernorm.weight', 'layers.4.enc_self_attn.layernorm.bias', 'layers.4.enc_self_attn.W_Q.weight', 'layers.4.enc_self_attn.W_Q.bias', 'layers.4.enc_self_attn.W_K.weight', 'layers.4.enc_self_attn.W_K.bias', 'layers.4.enc_self_attn.W_V.weight', 'layers.4.enc_self_attn.W_V.bias', 'layers.4.pos_ffn.fc.0.weight', 'layers.4.pos_ffn.fc.2.weight', 'layers.4.pos_ffn.layernorm.weight', 'layers.4.pos_ffn.layernorm.bias', 'layers.5.enc_self_attn.linear.weight', 'layers.5.enc_self_attn.linear.bias', 'layers.5.enc_self_attn.layernorm.weight', 'layers.5.enc_self_attn.layernorm.bias', 'layers.5.enc_self_attn.W_Q.weight', 'layers.5.enc_self_attn.W_Q.bias', 'layers.5.enc_self_attn.W_K.weight', 'layers.5.enc_self_attn.W_K.bias', 'layers.5.enc_self_attn.W_V.weight', 'layers.5.enc_self_attn.W_V.bias', 'layers.5.pos_ffn.fc.0.weight', 'layers.5.pos_ffn.fc.2.weight', 'layers.5.pos_ffn.layernorm.weight', 'layers.5.pos_ffn.layernorm.bias', 'layers.6.enc_self_attn.linear.weight', 'layers.6.enc_self_attn.linear.bias', 'layers.6.enc_self_attn.layernorm.weight', 'layers.6.enc_self_attn.layernorm.bias', 'layers.6.enc_self_attn.W_Q.weight', 'layers.6.enc_self_attn.W_Q.bias', 'layers.6.enc_self_attn.W_K.weight', 'layers.6.enc_self_attn.W_K.bias', 'layers.6.enc_self_attn.W_V.weight', 'layers.6.enc_self_attn.W_V.bias', 'layers.6.pos_ffn.fc.0.weight', 'layers.6.pos_ffn.fc.2.weight', 'layers.6.pos_ffn.layernorm.weight', 'layers.6.pos_ffn.layernorm.bias']
        elif self.pretrain_layer == 8:
            pretrained_parameters = ['embedding.tok_embed.weight', 'embedding.pos_embed.weight',
                                     'embedding.norm.weight', 'embedding.norm.bias',
                                     'layers.0.enc_self_attn.linear.weight', 'layers.0.enc_self_attn.linear.bias',
                                     'layers.0.enc_self_attn.layernorm.weight', 'layers.0.enc_self_attn.layernorm.bias',
                                     'layers.0.enc_self_attn.W_Q.weight', 'layers.0.enc_self_attn.W_Q.bias',
                                     'layers.0.enc_self_attn.W_K.weight', 'layers.0.enc_self_attn.W_K.bias',
                                     'layers.0.enc_self_attn.W_V.weight', 'layers.0.enc_self_attn.W_V.bias',
                                     'layers.0.pos_ffn.fc.0.weight', 'layers.0.pos_ffn.fc.2.weight',
                                     'layers.0.pos_ffn.layernorm.weight', 'layers.0.pos_ffn.layernorm.bias',
                                     'layers.1.enc_self_attn.linear.weight', 'layers.1.enc_self_attn.linear.bias',
                                     'layers.1.enc_self_attn.layernorm.weight', 'layers.1.enc_self_attn.layernorm.bias',
                                     'layers.1.enc_self_attn.W_Q.weight', 'layers.1.enc_self_attn.W_Q.bias',
                                     'layers.1.enc_self_attn.W_K.weight', 'layers.1.enc_self_attn.W_K.bias',
                                     'layers.1.enc_self_attn.W_V.weight', 'layers.1.enc_self_attn.W_V.bias',
                                     'layers.1.pos_ffn.fc.0.weight', 'layers.1.pos_ffn.fc.2.weight',
                                     'layers.1.pos_ffn.layernorm.weight', 'layers.1.pos_ffn.layernorm.bias',
                                     'layers.2.enc_self_attn.linear.weight', 'layers.2.enc_self_attn.linear.bias',
                                     'layers.2.enc_self_attn.layernorm.weight', 'layers.2.enc_self_attn.layernorm.bias',
                                     'layers.2.enc_self_attn.W_Q.weight', 'layers.2.enc_self_attn.W_Q.bias',
                                     'layers.2.enc_self_attn.W_K.weight', 'layers.2.enc_self_attn.W_K.bias',
                                     'layers.2.enc_self_attn.W_V.weight', 'layers.2.enc_self_attn.W_V.bias',
                                     'layers.2.pos_ffn.fc.0.weight', 'layers.2.pos_ffn.fc.2.weight',
                                     'layers.2.pos_ffn.layernorm.weight', 'layers.2.pos_ffn.layernorm.bias',
                                     'layers.3.enc_self_attn.linear.weight', 'layers.3.enc_self_attn.linear.bias',
                                     'layers.3.enc_self_attn.layernorm.weight', 'layers.3.enc_self_attn.layernorm.bias',
                                     'layers.3.enc_self_attn.W_Q.weight', 'layers.3.enc_self_attn.W_Q.bias',
                                     'layers.3.enc_self_attn.W_K.weight', 'layers.3.enc_self_attn.W_K.bias',
                                     'layers.3.enc_self_attn.W_V.weight', 'layers.3.enc_self_attn.W_V.bias',
                                     'layers.3.pos_ffn.fc.0.weight', 'layers.3.pos_ffn.fc.2.weight',
                                     'layers.3.pos_ffn.layernorm.weight', 'layers.3.pos_ffn.layernorm.bias',
                                     'layers.4.enc_self_attn.linear.weight', 'layers.4.enc_self_attn.linear.bias',
                                     'layers.4.enc_self_attn.layernorm.weight', 'layers.4.enc_self_attn.layernorm.bias',
                                     'layers.4.enc_self_attn.W_Q.weight', 'layers.4.enc_self_attn.W_Q.bias',
                                     'layers.4.enc_self_attn.W_K.weight', 'layers.4.enc_self_attn.W_K.bias',
                                     'layers.4.enc_self_attn.W_V.weight', 'layers.4.enc_self_attn.W_V.bias',
                                     'layers.4.pos_ffn.fc.0.weight', 'layers.4.pos_ffn.fc.2.weight',
                                     'layers.4.pos_ffn.layernorm.weight', 'layers.4.pos_ffn.layernorm.bias',
                                     'layers.5.enc_self_attn.linear.weight', 'layers.5.enc_self_attn.linear.bias',
                                     'layers.5.enc_self_attn.layernorm.weight', 'layers.5.enc_self_attn.layernorm.bias',
                                     'layers.5.enc_self_attn.W_Q.weight', 'layers.5.enc_self_attn.W_Q.bias',
                                     'layers.5.enc_self_attn.W_K.weight', 'layers.5.enc_self_attn.W_K.bias',
                                     'layers.5.enc_self_attn.W_V.weight', 'layers.5.enc_self_attn.W_V.bias',
                                     'layers.5.pos_ffn.fc.0.weight', 'layers.5.pos_ffn.fc.2.weight',
                                     'layers.5.pos_ffn.layernorm.weight', 'layers.5.pos_ffn.layernorm.bias',
                                     'layers.6.enc_self_attn.linear.weight', 'layers.6.enc_self_attn.linear.bias',
                                     'layers.6.enc_self_attn.layernorm.weight', 'layers.6.enc_self_attn.layernorm.bias',
                                     'layers.6.enc_self_attn.W_Q.weight', 'layers.6.enc_self_attn.W_Q.bias',
                                     'layers.6.enc_self_attn.W_K.weight', 'layers.6.enc_self_attn.W_K.bias',
                                     'layers.6.enc_self_attn.W_V.weight', 'layers.6.enc_self_attn.W_V.bias',
                                     'layers.6.pos_ffn.fc.0.weight', 'layers.6.pos_ffn.fc.2.weight',
                                     'layers.6.pos_ffn.layernorm.weight', 'layers.6.pos_ffn.layernorm.bias',
                                     'layers.7.enc_self_attn.linear.weight', 'layers.7.enc_self_attn.linear.bias',
                                     'layers.7.enc_self_attn.layernorm.weight', 'layers.7.enc_self_attn.layernorm.bias',
                                     'layers.7.enc_self_attn.W_Q.weight', 'layers.7.enc_self_attn.W_Q.bias',
                                     'layers.7.enc_self_attn.W_K.weight', 'layers.7.enc_self_attn.W_K.bias',
                                     'layers.7.enc_self_attn.W_V.weight', 'layers.7.enc_self_attn.W_V.bias',
                                     'layers.7.pos_ffn.fc.0.weight', 'layers.7.pos_ffn.fc.2.weight',
                                     'layers.7.pos_ffn.layernorm.weight', 'layers.7.pos_ffn.layernorm.bias']
        elif self.pretrain_layer == 9:
            pretrained_parameters = ['embedding.tok_embed.weight', 'embedding.pos_embed.weight',
                                     'embedding.norm.weight', 'embedding.norm.bias',
                                     'layers.0.enc_self_attn.linear.weight', 'layers.0.enc_self_attn.linear.bias',
                                     'layers.0.enc_self_attn.layernorm.weight', 'layers.0.enc_self_attn.layernorm.bias',
                                     'layers.0.enc_self_attn.W_Q.weight', 'layers.0.enc_self_attn.W_Q.bias',
                                     'layers.0.enc_self_attn.W_K.weight', 'layers.0.enc_self_attn.W_K.bias',
                                     'layers.0.enc_self_attn.W_V.weight', 'layers.0.enc_self_attn.W_V.bias',
                                     'layers.0.pos_ffn.fc.0.weight', 'layers.0.pos_ffn.fc.2.weight',
                                     'layers.0.pos_ffn.layernorm.weight', 'layers.0.pos_ffn.layernorm.bias',
                                     'layers.1.enc_self_attn.linear.weight', 'layers.1.enc_self_attn.linear.bias',
                                     'layers.1.enc_self_attn.layernorm.weight', 'layers.1.enc_self_attn.layernorm.bias',
                                     'layers.1.enc_self_attn.W_Q.weight', 'layers.1.enc_self_attn.W_Q.bias',
                                     'layers.1.enc_self_attn.W_K.weight', 'layers.1.enc_self_attn.W_K.bias',
                                     'layers.1.enc_self_attn.W_V.weight', 'layers.1.enc_self_attn.W_V.bias',
                                     'layers.1.pos_ffn.fc.0.weight', 'layers.1.pos_ffn.fc.2.weight',
                                     'layers.1.pos_ffn.layernorm.weight', 'layers.1.pos_ffn.layernorm.bias',
                                     'layers.2.enc_self_attn.linear.weight', 'layers.2.enc_self_attn.linear.bias',
                                     'layers.2.enc_self_attn.layernorm.weight', 'layers.2.enc_self_attn.layernorm.bias',
                                     'layers.2.enc_self_attn.W_Q.weight', 'layers.2.enc_self_attn.W_Q.bias',
                                     'layers.2.enc_self_attn.W_K.weight', 'layers.2.enc_self_attn.W_K.bias',
                                     'layers.2.enc_self_attn.W_V.weight', 'layers.2.enc_self_attn.W_V.bias',
                                     'layers.2.pos_ffn.fc.0.weight', 'layers.2.pos_ffn.fc.2.weight',
                                     'layers.2.pos_ffn.layernorm.weight', 'layers.2.pos_ffn.layernorm.bias',
                                     'layers.3.enc_self_attn.linear.weight', 'layers.3.enc_self_attn.linear.bias',
                                     'layers.3.enc_self_attn.layernorm.weight', 'layers.3.enc_self_attn.layernorm.bias',
                                     'layers.3.enc_self_attn.W_Q.weight', 'layers.3.enc_self_attn.W_Q.bias',
                                     'layers.3.enc_self_attn.W_K.weight', 'layers.3.enc_self_attn.W_K.bias',
                                     'layers.3.enc_self_attn.W_V.weight', 'layers.3.enc_self_attn.W_V.bias',
                                     'layers.3.pos_ffn.fc.0.weight', 'layers.3.pos_ffn.fc.2.weight',
                                     'layers.3.pos_ffn.layernorm.weight', 'layers.3.pos_ffn.layernorm.bias',
                                     'layers.4.enc_self_attn.linear.weight', 'layers.4.enc_self_attn.linear.bias',
                                     'layers.4.enc_self_attn.layernorm.weight', 'layers.4.enc_self_attn.layernorm.bias',
                                     'layers.4.enc_self_attn.W_Q.weight', 'layers.4.enc_self_attn.W_Q.bias',
                                     'layers.4.enc_self_attn.W_K.weight', 'layers.4.enc_self_attn.W_K.bias',
                                     'layers.4.enc_self_attn.W_V.weight', 'layers.4.enc_self_attn.W_V.bias',
                                     'layers.4.pos_ffn.fc.0.weight', 'layers.4.pos_ffn.fc.2.weight',
                                     'layers.4.pos_ffn.layernorm.weight', 'layers.4.pos_ffn.layernorm.bias',
                                     'layers.5.enc_self_attn.linear.weight', 'layers.5.enc_self_attn.linear.bias',
                                     'layers.5.enc_self_attn.layernorm.weight', 'layers.5.enc_self_attn.layernorm.bias',
                                     'layers.5.enc_self_attn.W_Q.weight', 'layers.5.enc_self_attn.W_Q.bias',
                                     'layers.5.enc_self_attn.W_K.weight', 'layers.5.enc_self_attn.W_K.bias',
                                     'layers.5.enc_self_attn.W_V.weight', 'layers.5.enc_self_attn.W_V.bias',
                                     'layers.5.pos_ffn.fc.0.weight', 'layers.5.pos_ffn.fc.2.weight',
                                     'layers.5.pos_ffn.layernorm.weight', 'layers.5.pos_ffn.layernorm.bias',
                                     'layers.6.enc_self_attn.linear.weight', 'layers.6.enc_self_attn.linear.bias',
                                     'layers.6.enc_self_attn.layernorm.weight', 'layers.6.enc_self_attn.layernorm.bias',
                                     'layers.6.enc_self_attn.W_Q.weight', 'layers.6.enc_self_attn.W_Q.bias',
                                     'layers.6.enc_self_attn.W_K.weight', 'layers.6.enc_self_attn.W_K.bias',
                                     'layers.6.enc_self_attn.W_V.weight', 'layers.6.enc_self_attn.W_V.bias',
                                     'layers.6.pos_ffn.fc.0.weight', 'layers.6.pos_ffn.fc.2.weight',
                                     'layers.6.pos_ffn.layernorm.weight', 'layers.6.pos_ffn.layernorm.bias',
                                     'layers.7.enc_self_attn.linear.weight', 'layers.7.enc_self_attn.linear.bias',
                                     'layers.7.enc_self_attn.layernorm.weight', 'layers.7.enc_self_attn.layernorm.bias',
                                     'layers.7.enc_self_attn.W_Q.weight', 'layers.7.enc_self_attn.W_Q.bias',
                                     'layers.7.enc_self_attn.W_K.weight', 'layers.7.enc_self_attn.W_K.bias',
                                     'layers.7.enc_self_attn.W_V.weight', 'layers.7.enc_self_attn.W_V.bias',
                                     'layers.7.pos_ffn.fc.0.weight', 'layers.7.pos_ffn.fc.2.weight',
                                     'layers.7.pos_ffn.layernorm.weight', 'layers.7.pos_ffn.layernorm.bias',
                                     'layers.8.enc_self_attn.linear.weight', 'layers.8.enc_self_attn.linear.bias',
                                     'layers.8.enc_self_attn.layernorm.weight', 'layers.8.enc_self_attn.layernorm.bias',
                                     'layers.8.enc_self_attn.W_Q.weight', 'layers.8.enc_self_attn.W_Q.bias',
                                     'layers.8.enc_self_attn.W_K.weight', 'layers.8.enc_self_attn.W_K.bias',
                                     'layers.8.enc_self_attn.W_V.weight', 'layers.8.enc_self_attn.W_V.bias',
                                     'layers.8.pos_ffn.fc.0.weight', 'layers.8.pos_ffn.fc.2.weight',
                                     'layers.8.pos_ffn.layernorm.weight', 'layers.8.pos_ffn.layernorm.bias']
        elif self.pretrain_layer == 10:
            pretrained_parameters = ['embedding.tok_embed.weight', 'embedding.pos_embed.weight',
                                     'embedding.norm.weight', 'embedding.norm.bias',
                                     'layers.0.enc_self_attn.linear.weight', 'layers.0.enc_self_attn.linear.bias',
                                     'layers.0.enc_self_attn.layernorm.weight', 'layers.0.enc_self_attn.layernorm.bias',
                                     'layers.0.enc_self_attn.W_Q.weight', 'layers.0.enc_self_attn.W_Q.bias',
                                     'layers.0.enc_self_attn.W_K.weight', 'layers.0.enc_self_attn.W_K.bias',
                                     'layers.0.enc_self_attn.W_V.weight', 'layers.0.enc_self_attn.W_V.bias',
                                     'layers.0.pos_ffn.fc.0.weight', 'layers.0.pos_ffn.fc.2.weight',
                                     'layers.0.pos_ffn.layernorm.weight', 'layers.0.pos_ffn.layernorm.bias',
                                     'layers.1.enc_self_attn.linear.weight', 'layers.1.enc_self_attn.linear.bias',
                                     'layers.1.enc_self_attn.layernorm.weight', 'layers.1.enc_self_attn.layernorm.bias',
                                     'layers.1.enc_self_attn.W_Q.weight', 'layers.1.enc_self_attn.W_Q.bias',
                                     'layers.1.enc_self_attn.W_K.weight', 'layers.1.enc_self_attn.W_K.bias',
                                     'layers.1.enc_self_attn.W_V.weight', 'layers.1.enc_self_attn.W_V.bias',
                                     'layers.1.pos_ffn.fc.0.weight', 'layers.1.pos_ffn.fc.2.weight',
                                     'layers.1.pos_ffn.layernorm.weight', 'layers.1.pos_ffn.layernorm.bias',
                                     'layers.2.enc_self_attn.linear.weight', 'layers.2.enc_self_attn.linear.bias',
                                     'layers.2.enc_self_attn.layernorm.weight', 'layers.2.enc_self_attn.layernorm.bias',
                                     'layers.2.enc_self_attn.W_Q.weight', 'layers.2.enc_self_attn.W_Q.bias',
                                     'layers.2.enc_self_attn.W_K.weight', 'layers.2.enc_self_attn.W_K.bias',
                                     'layers.2.enc_self_attn.W_V.weight', 'layers.2.enc_self_attn.W_V.bias',
                                     'layers.2.pos_ffn.fc.0.weight', 'layers.2.pos_ffn.fc.2.weight',
                                     'layers.2.pos_ffn.layernorm.weight', 'layers.2.pos_ffn.layernorm.bias',
                                     'layers.3.enc_self_attn.linear.weight', 'layers.3.enc_self_attn.linear.bias',
                                     'layers.3.enc_self_attn.layernorm.weight', 'layers.3.enc_self_attn.layernorm.bias',
                                     'layers.3.enc_self_attn.W_Q.weight', 'layers.3.enc_self_attn.W_Q.bias',
                                     'layers.3.enc_self_attn.W_K.weight', 'layers.3.enc_self_attn.W_K.bias',
                                     'layers.3.enc_self_attn.W_V.weight', 'layers.3.enc_self_attn.W_V.bias',
                                     'layers.3.pos_ffn.fc.0.weight', 'layers.3.pos_ffn.fc.2.weight',
                                     'layers.3.pos_ffn.layernorm.weight', 'layers.3.pos_ffn.layernorm.bias',
                                     'layers.4.enc_self_attn.linear.weight', 'layers.4.enc_self_attn.linear.bias',
                                     'layers.4.enc_self_attn.layernorm.weight', 'layers.4.enc_self_attn.layernorm.bias',
                                     'layers.4.enc_self_attn.W_Q.weight', 'layers.4.enc_self_attn.W_Q.bias',
                                     'layers.4.enc_self_attn.W_K.weight', 'layers.4.enc_self_attn.W_K.bias',
                                     'layers.4.enc_self_attn.W_V.weight', 'layers.4.enc_self_attn.W_V.bias',
                                     'layers.4.pos_ffn.fc.0.weight', 'layers.4.pos_ffn.fc.2.weight',
                                     'layers.4.pos_ffn.layernorm.weight', 'layers.4.pos_ffn.layernorm.bias',
                                     'layers.5.enc_self_attn.linear.weight', 'layers.5.enc_self_attn.linear.bias',
                                     'layers.5.enc_self_attn.layernorm.weight', 'layers.5.enc_self_attn.layernorm.bias',
                                     'layers.5.enc_self_attn.W_Q.weight', 'layers.5.enc_self_attn.W_Q.bias',
                                     'layers.5.enc_self_attn.W_K.weight', 'layers.5.enc_self_attn.W_K.bias',
                                     'layers.5.enc_self_attn.W_V.weight', 'layers.5.enc_self_attn.W_V.bias',
                                     'layers.5.pos_ffn.fc.0.weight', 'layers.5.pos_ffn.fc.2.weight',
                                     'layers.5.pos_ffn.layernorm.weight', 'layers.5.pos_ffn.layernorm.bias',
                                     'layers.6.enc_self_attn.linear.weight', 'layers.6.enc_self_attn.linear.bias',
                                     'layers.6.enc_self_attn.layernorm.weight', 'layers.6.enc_self_attn.layernorm.bias',
                                     'layers.6.enc_self_attn.W_Q.weight', 'layers.6.enc_self_attn.W_Q.bias',
                                     'layers.6.enc_self_attn.W_K.weight', 'layers.6.enc_self_attn.W_K.bias',
                                     'layers.6.enc_self_attn.W_V.weight', 'layers.6.enc_self_attn.W_V.bias',
                                     'layers.6.pos_ffn.fc.0.weight', 'layers.6.pos_ffn.fc.2.weight',
                                     'layers.6.pos_ffn.layernorm.weight', 'layers.6.pos_ffn.layernorm.bias',
                                     'layers.7.enc_self_attn.linear.weight', 'layers.7.enc_self_attn.linear.bias',
                                     'layers.7.enc_self_attn.layernorm.weight', 'layers.7.enc_self_attn.layernorm.bias',
                                     'layers.7.enc_self_attn.W_Q.weight', 'layers.7.enc_self_attn.W_Q.bias',
                                     'layers.7.enc_self_attn.W_K.weight', 'layers.7.enc_self_attn.W_K.bias',
                                     'layers.7.enc_self_attn.W_V.weight', 'layers.7.enc_self_attn.W_V.bias',
                                     'layers.7.pos_ffn.fc.0.weight', 'layers.7.pos_ffn.fc.2.weight',
                                     'layers.7.pos_ffn.layernorm.weight', 'layers.7.pos_ffn.layernorm.bias',
                                     'layers.8.enc_self_attn.linear.weight', 'layers.8.enc_self_attn.linear.bias',
                                     'layers.8.enc_self_attn.layernorm.weight', 'layers.8.enc_self_attn.layernorm.bias',
                                     'layers.8.enc_self_attn.W_Q.weight', 'layers.8.enc_self_attn.W_Q.bias',
                                     'layers.8.enc_self_attn.W_K.weight', 'layers.8.enc_self_attn.W_K.bias',
                                     'layers.8.enc_self_attn.W_V.weight', 'layers.8.enc_self_attn.W_V.bias',
                                     'layers.8.pos_ffn.fc.0.weight', 'layers.8.pos_ffn.fc.2.weight',
                                     'layers.8.pos_ffn.layernorm.weight', 'layers.8.pos_ffn.layernorm.bias',
                                     'layers.9.enc_self_attn.linear.weight', 'layers.9.enc_self_attn.linear.bias',
                                     'layers.9.enc_self_attn.layernorm.weight', 'layers.9.enc_self_attn.layernorm.bias',
                                     'layers.9.enc_self_attn.W_Q.weight', 'layers.9.enc_self_attn.W_Q.bias',
                                     'layers.9.enc_self_attn.W_K.weight', 'layers.9.enc_self_attn.W_K.bias',
                                     'layers.9.enc_self_attn.W_V.weight', 'layers.9.enc_self_attn.W_V.bias',
                                     'layers.9.pos_ffn.fc.0.weight', 'layers.9.pos_ffn.fc.2.weight',
                                     'layers.9.pos_ffn.layernorm.weight', 'layers.9.pos_ffn.layernorm.bias']
        elif self.pretrain_layer == 11:
            pretrained_parameters = ['embedding.tok_embed.weight', 'embedding.pos_embed.weight',
                                     'embedding.norm.weight', 'embedding.norm.bias',
                                     'layers.0.enc_self_attn.linear.weight', 'layers.0.enc_self_attn.linear.bias',
                                     'layers.0.enc_self_attn.layernorm.weight', 'layers.0.enc_self_attn.layernorm.bias',
                                     'layers.0.enc_self_attn.W_Q.weight', 'layers.0.enc_self_attn.W_Q.bias',
                                     'layers.0.enc_self_attn.W_K.weight', 'layers.0.enc_self_attn.W_K.bias',
                                     'layers.0.enc_self_attn.W_V.weight', 'layers.0.enc_self_attn.W_V.bias',
                                     'layers.0.pos_ffn.fc.0.weight', 'layers.0.pos_ffn.fc.2.weight',
                                     'layers.0.pos_ffn.layernorm.weight', 'layers.0.pos_ffn.layernorm.bias',
                                     'layers.1.enc_self_attn.linear.weight', 'layers.1.enc_self_attn.linear.bias',
                                     'layers.1.enc_self_attn.layernorm.weight', 'layers.1.enc_self_attn.layernorm.bias',
                                     'layers.1.enc_self_attn.W_Q.weight', 'layers.1.enc_self_attn.W_Q.bias',
                                     'layers.1.enc_self_attn.W_K.weight', 'layers.1.enc_self_attn.W_K.bias',
                                     'layers.1.enc_self_attn.W_V.weight', 'layers.1.enc_self_attn.W_V.bias',
                                     'layers.1.pos_ffn.fc.0.weight', 'layers.1.pos_ffn.fc.2.weight',
                                     'layers.1.pos_ffn.layernorm.weight', 'layers.1.pos_ffn.layernorm.bias',
                                     'layers.2.enc_self_attn.linear.weight', 'layers.2.enc_self_attn.linear.bias',
                                     'layers.2.enc_self_attn.layernorm.weight', 'layers.2.enc_self_attn.layernorm.bias',
                                     'layers.2.enc_self_attn.W_Q.weight', 'layers.2.enc_self_attn.W_Q.bias',
                                     'layers.2.enc_self_attn.W_K.weight', 'layers.2.enc_self_attn.W_K.bias',
                                     'layers.2.enc_self_attn.W_V.weight', 'layers.2.enc_self_attn.W_V.bias',
                                     'layers.2.pos_ffn.fc.0.weight', 'layers.2.pos_ffn.fc.2.weight',
                                     'layers.2.pos_ffn.layernorm.weight', 'layers.2.pos_ffn.layernorm.bias',
                                     'layers.3.enc_self_attn.linear.weight', 'layers.3.enc_self_attn.linear.bias',
                                     'layers.3.enc_self_attn.layernorm.weight', 'layers.3.enc_self_attn.layernorm.bias',
                                     'layers.3.enc_self_attn.W_Q.weight', 'layers.3.enc_self_attn.W_Q.bias',
                                     'layers.3.enc_self_attn.W_K.weight', 'layers.3.enc_self_attn.W_K.bias',
                                     'layers.3.enc_self_attn.W_V.weight', 'layers.3.enc_self_attn.W_V.bias',
                                     'layers.3.pos_ffn.fc.0.weight', 'layers.3.pos_ffn.fc.2.weight',
                                     'layers.3.pos_ffn.layernorm.weight', 'layers.3.pos_ffn.layernorm.bias',
                                     'layers.4.enc_self_attn.linear.weight', 'layers.4.enc_self_attn.linear.bias',
                                     'layers.4.enc_self_attn.layernorm.weight', 'layers.4.enc_self_attn.layernorm.bias',
                                     'layers.4.enc_self_attn.W_Q.weight', 'layers.4.enc_self_attn.W_Q.bias',
                                     'layers.4.enc_self_attn.W_K.weight', 'layers.4.enc_self_attn.W_K.bias',
                                     'layers.4.enc_self_attn.W_V.weight', 'layers.4.enc_self_attn.W_V.bias',
                                     'layers.4.pos_ffn.fc.0.weight', 'layers.4.pos_ffn.fc.2.weight',
                                     'layers.4.pos_ffn.layernorm.weight', 'layers.4.pos_ffn.layernorm.bias',
                                     'layers.5.enc_self_attn.linear.weight', 'layers.5.enc_self_attn.linear.bias',
                                     'layers.5.enc_self_attn.layernorm.weight', 'layers.5.enc_self_attn.layernorm.bias',
                                     'layers.5.enc_self_attn.W_Q.weight', 'layers.5.enc_self_attn.W_Q.bias',
                                     'layers.5.enc_self_attn.W_K.weight', 'layers.5.enc_self_attn.W_K.bias',
                                     'layers.5.enc_self_attn.W_V.weight', 'layers.5.enc_self_attn.W_V.bias',
                                     'layers.5.pos_ffn.fc.0.weight', 'layers.5.pos_ffn.fc.2.weight',
                                     'layers.5.pos_ffn.layernorm.weight', 'layers.5.pos_ffn.layernorm.bias',
                                     'layers.6.enc_self_attn.linear.weight', 'layers.6.enc_self_attn.linear.bias',
                                     'layers.6.enc_self_attn.layernorm.weight', 'layers.6.enc_self_attn.layernorm.bias',
                                     'layers.6.enc_self_attn.W_Q.weight', 'layers.6.enc_self_attn.W_Q.bias',
                                     'layers.6.enc_self_attn.W_K.weight', 'layers.6.enc_self_attn.W_K.bias',
                                     'layers.6.enc_self_attn.W_V.weight', 'layers.6.enc_self_attn.W_V.bias',
                                     'layers.6.pos_ffn.fc.0.weight', 'layers.6.pos_ffn.fc.2.weight',
                                     'layers.6.pos_ffn.layernorm.weight', 'layers.6.pos_ffn.layernorm.bias',
                                     'layers.7.enc_self_attn.linear.weight', 'layers.7.enc_self_attn.linear.bias',
                                     'layers.7.enc_self_attn.layernorm.weight', 'layers.7.enc_self_attn.layernorm.bias',
                                     'layers.7.enc_self_attn.W_Q.weight', 'layers.7.enc_self_attn.W_Q.bias',
                                     'layers.7.enc_self_attn.W_K.weight', 'layers.7.enc_self_attn.W_K.bias',
                                     'layers.7.enc_self_attn.W_V.weight', 'layers.7.enc_self_attn.W_V.bias',
                                     'layers.7.pos_ffn.fc.0.weight', 'layers.7.pos_ffn.fc.2.weight',
                                     'layers.7.pos_ffn.layernorm.weight', 'layers.7.pos_ffn.layernorm.bias',
                                     'layers.8.enc_self_attn.linear.weight', 'layers.8.enc_self_attn.linear.bias',
                                     'layers.8.enc_self_attn.layernorm.weight', 'layers.8.enc_self_attn.layernorm.bias',
                                     'layers.8.enc_self_attn.W_Q.weight', 'layers.8.enc_self_attn.W_Q.bias',
                                     'layers.8.enc_self_attn.W_K.weight', 'layers.8.enc_self_attn.W_K.bias',
                                     'layers.8.enc_self_attn.W_V.weight', 'layers.8.enc_self_attn.W_V.bias',
                                     'layers.8.pos_ffn.fc.0.weight', 'layers.8.pos_ffn.fc.2.weight',
                                     'layers.8.pos_ffn.layernorm.weight', 'layers.8.pos_ffn.layernorm.bias',
                                     'layers.9.enc_self_attn.linear.weight', 'layers.9.enc_self_attn.linear.bias',
                                     'layers.9.enc_self_attn.layernorm.weight', 'layers.9.enc_self_attn.layernorm.bias',
                                     'layers.9.enc_self_attn.W_Q.weight', 'layers.9.enc_self_attn.W_Q.bias',
                                     'layers.9.enc_self_attn.W_K.weight', 'layers.9.enc_self_attn.W_K.bias',
                                     'layers.9.enc_self_attn.W_V.weight', 'layers.9.enc_self_attn.W_V.bias',
                                     'layers.9.pos_ffn.fc.0.weight', 'layers.9.pos_ffn.fc.2.weight',
                                     'layers.9.pos_ffn.layernorm.weight', 'layers.9.pos_ffn.layernorm.bias',
                                     'layers.10.enc_self_attn.linear.weight', 'layers.10.enc_self_attn.linear.bias',
                                     'layers.10.enc_self_attn.layernorm.weight',
                                     'layers.10.enc_self_attn.layernorm.bias', 'layers.10.enc_self_attn.W_Q.weight',
                                     'layers.10.enc_self_attn.W_Q.bias', 'layers.10.enc_self_attn.W_K.weight',
                                     'layers.10.enc_self_attn.W_K.bias', 'layers.10.enc_self_attn.W_V.weight',
                                     'layers.10.enc_self_attn.W_V.bias', 'layers.10.pos_ffn.fc.0.weight',
                                     'layers.10.pos_ffn.fc.2.weight', 'layers.10.pos_ffn.layernorm.weight',
                                     'layers.10.pos_ffn.layernorm.bias']
        elif self.pretrain_layer == 12:
            pretrained_parameters = ['embedding.tok_embed.weight', 'embedding.pos_embed.weight',
                                     'embedding.norm.weight', 'embedding.norm.bias',
                                     'layers.0.enc_self_attn.linear.weight', 'layers.0.enc_self_attn.linear.bias',
                                     'layers.0.enc_self_attn.layernorm.weight', 'layers.0.enc_self_attn.layernorm.bias',
                                     'layers.0.enc_self_attn.W_Q.weight', 'layers.0.enc_self_attn.W_Q.bias',
                                     'layers.0.enc_self_attn.W_K.weight', 'layers.0.enc_self_attn.W_K.bias',
                                     'layers.0.enc_self_attn.W_V.weight', 'layers.0.enc_self_attn.W_V.bias',
                                     'layers.0.pos_ffn.fc.0.weight', 'layers.0.pos_ffn.fc.2.weight',
                                     'layers.0.pos_ffn.layernorm.weight', 'layers.0.pos_ffn.layernorm.bias',
                                     'layers.1.enc_self_attn.linear.weight', 'layers.1.enc_self_attn.linear.bias',
                                     'layers.1.enc_self_attn.layernorm.weight', 'layers.1.enc_self_attn.layernorm.bias',
                                     'layers.1.enc_self_attn.W_Q.weight', 'layers.1.enc_self_attn.W_Q.bias',
                                     'layers.1.enc_self_attn.W_K.weight', 'layers.1.enc_self_attn.W_K.bias',
                                     'layers.1.enc_self_attn.W_V.weight', 'layers.1.enc_self_attn.W_V.bias',
                                     'layers.1.pos_ffn.fc.0.weight', 'layers.1.pos_ffn.fc.2.weight',
                                     'layers.1.pos_ffn.layernorm.weight', 'layers.1.pos_ffn.layernorm.bias',
                                     'layers.2.enc_self_attn.linear.weight', 'layers.2.enc_self_attn.linear.bias',
                                     'layers.2.enc_self_attn.layernorm.weight', 'layers.2.enc_self_attn.layernorm.bias',
                                     'layers.2.enc_self_attn.W_Q.weight', 'layers.2.enc_self_attn.W_Q.bias',
                                     'layers.2.enc_self_attn.W_K.weight', 'layers.2.enc_self_attn.W_K.bias',
                                     'layers.2.enc_self_attn.W_V.weight', 'layers.2.enc_self_attn.W_V.bias',
                                     'layers.2.pos_ffn.fc.0.weight', 'layers.2.pos_ffn.fc.2.weight',
                                     'layers.2.pos_ffn.layernorm.weight', 'layers.2.pos_ffn.layernorm.bias',
                                     'layers.3.enc_self_attn.linear.weight', 'layers.3.enc_self_attn.linear.bias',
                                     'layers.3.enc_self_attn.layernorm.weight', 'layers.3.enc_self_attn.layernorm.bias',
                                     'layers.3.enc_self_attn.W_Q.weight', 'layers.3.enc_self_attn.W_Q.bias',
                                     'layers.3.enc_self_attn.W_K.weight', 'layers.3.enc_self_attn.W_K.bias',
                                     'layers.3.enc_self_attn.W_V.weight', 'layers.3.enc_self_attn.W_V.bias',
                                     'layers.3.pos_ffn.fc.0.weight', 'layers.3.pos_ffn.fc.2.weight',
                                     'layers.3.pos_ffn.layernorm.weight', 'layers.3.pos_ffn.layernorm.bias',
                                     'layers.4.enc_self_attn.linear.weight', 'layers.4.enc_self_attn.linear.bias',
                                     'layers.4.enc_self_attn.layernorm.weight', 'layers.4.enc_self_attn.layernorm.bias',
                                     'layers.4.enc_self_attn.W_Q.weight', 'layers.4.enc_self_attn.W_Q.bias',
                                     'layers.4.enc_self_attn.W_K.weight', 'layers.4.enc_self_attn.W_K.bias',
                                     'layers.4.enc_self_attn.W_V.weight', 'layers.4.enc_self_attn.W_V.bias',
                                     'layers.4.pos_ffn.fc.0.weight', 'layers.4.pos_ffn.fc.2.weight',
                                     'layers.4.pos_ffn.layernorm.weight', 'layers.4.pos_ffn.layernorm.bias',
                                     'layers.5.enc_self_attn.linear.weight', 'layers.5.enc_self_attn.linear.bias',
                                     'layers.5.enc_self_attn.layernorm.weight', 'layers.5.enc_self_attn.layernorm.bias',
                                     'layers.5.enc_self_attn.W_Q.weight', 'layers.5.enc_self_attn.W_Q.bias',
                                     'layers.5.enc_self_attn.W_K.weight', 'layers.5.enc_self_attn.W_K.bias',
                                     'layers.5.enc_self_attn.W_V.weight', 'layers.5.enc_self_attn.W_V.bias',
                                     'layers.5.pos_ffn.fc.0.weight', 'layers.5.pos_ffn.fc.2.weight',
                                     'layers.5.pos_ffn.layernorm.weight', 'layers.5.pos_ffn.layernorm.bias',
                                     'layers.6.enc_self_attn.linear.weight', 'layers.6.enc_self_attn.linear.bias',
                                     'layers.6.enc_self_attn.layernorm.weight', 'layers.6.enc_self_attn.layernorm.bias',
                                     'layers.6.enc_self_attn.W_Q.weight', 'layers.6.enc_self_attn.W_Q.bias',
                                     'layers.6.enc_self_attn.W_K.weight', 'layers.6.enc_self_attn.W_K.bias',
                                     'layers.6.enc_self_attn.W_V.weight', 'layers.6.enc_self_attn.W_V.bias',
                                     'layers.6.pos_ffn.fc.0.weight', 'layers.6.pos_ffn.fc.2.weight',
                                     'layers.6.pos_ffn.layernorm.weight', 'layers.6.pos_ffn.layernorm.bias',
                                     'layers.7.enc_self_attn.linear.weight', 'layers.7.enc_self_attn.linear.bias',
                                     'layers.7.enc_self_attn.layernorm.weight', 'layers.7.enc_self_attn.layernorm.bias',
                                     'layers.7.enc_self_attn.W_Q.weight', 'layers.7.enc_self_attn.W_Q.bias',
                                     'layers.7.enc_self_attn.W_K.weight', 'layers.7.enc_self_attn.W_K.bias',
                                     'layers.7.enc_self_attn.W_V.weight', 'layers.7.enc_self_attn.W_V.bias',
                                     'layers.7.pos_ffn.fc.0.weight', 'layers.7.pos_ffn.fc.2.weight',
                                     'layers.7.pos_ffn.layernorm.weight', 'layers.7.pos_ffn.layernorm.bias',
                                     'layers.8.enc_self_attn.linear.weight', 'layers.8.enc_self_attn.linear.bias',
                                     'layers.8.enc_self_attn.layernorm.weight', 'layers.8.enc_self_attn.layernorm.bias',
                                     'layers.8.enc_self_attn.W_Q.weight', 'layers.8.enc_self_attn.W_Q.bias',
                                     'layers.8.enc_self_attn.W_K.weight', 'layers.8.enc_self_attn.W_K.bias',
                                     'layers.8.enc_self_attn.W_V.weight', 'layers.8.enc_self_attn.W_V.bias',
                                     'layers.8.pos_ffn.fc.0.weight', 'layers.8.pos_ffn.fc.2.weight',
                                     'layers.8.pos_ffn.layernorm.weight', 'layers.8.pos_ffn.layernorm.bias',
                                     'layers.9.enc_self_attn.linear.weight', 'layers.9.enc_self_attn.linear.bias',
                                     'layers.9.enc_self_attn.layernorm.weight', 'layers.9.enc_self_attn.layernorm.bias',
                                     'layers.9.enc_self_attn.W_Q.weight', 'layers.9.enc_self_attn.W_Q.bias',
                                     'layers.9.enc_self_attn.W_K.weight', 'layers.9.enc_self_attn.W_K.bias',
                                     'layers.9.enc_self_attn.W_V.weight', 'layers.9.enc_self_attn.W_V.bias',
                                     'layers.9.pos_ffn.fc.0.weight', 'layers.9.pos_ffn.fc.2.weight',
                                     'layers.9.pos_ffn.layernorm.weight', 'layers.9.pos_ffn.layernorm.bias',
                                     'layers.10.enc_self_attn.linear.weight', 'layers.10.enc_self_attn.linear.bias',
                                     'layers.10.enc_self_attn.layernorm.weight',
                                     'layers.10.enc_self_attn.layernorm.bias', 'layers.10.enc_self_attn.W_Q.weight',
                                     'layers.10.enc_self_attn.W_Q.bias', 'layers.10.enc_self_attn.W_K.weight',
                                     'layers.10.enc_self_attn.W_K.bias', 'layers.10.enc_self_attn.W_V.weight',
                                     'layers.10.enc_self_attn.W_V.bias', 'layers.10.pos_ffn.fc.0.weight',
                                     'layers.10.pos_ffn.fc.2.weight', 'layers.10.pos_ffn.layernorm.weight',
                                     'layers.10.pos_ffn.layernorm.bias', 'layers.11.enc_self_attn.linear.weight',
                                     'layers.11.enc_self_attn.linear.bias', 'layers.11.enc_self_attn.layernorm.weight',
                                     'layers.11.enc_self_attn.layernorm.bias', 'layers.11.enc_self_attn.W_Q.weight',
                                     'layers.11.enc_self_attn.W_Q.bias', 'layers.11.enc_self_attn.W_K.weight',
                                     'layers.11.enc_self_attn.W_K.bias', 'layers.11.enc_self_attn.W_V.weight',
                                     'layers.11.enc_self_attn.W_V.bias', 'layers.11.pos_ffn.fc.0.weight',
                                     'layers.11.pos_ffn.fc.2.weight', 'layers.11.pos_ffn.layernorm.weight',
                                     'layers.11.pos_ffn.layernorm.bias']
        elif self.pretrain_layer == 'all_6layer':
            pretrained_parameters = ['embedding.tok_embed.weight', 'embedding.pos_embed.weight', 'embedding.norm.weight', 'embedding.norm.bias', 'layers.0.enc_self_attn.linear.weight', 'layers.0.enc_self_attn.linear.bias', 'layers.0.enc_self_attn.layernorm.weight', 'layers.0.enc_self_attn.layernorm.bias', 'layers.0.enc_self_attn.W_Q.weight', 'layers.0.enc_self_attn.W_Q.bias', 'layers.0.enc_self_attn.W_K.weight', 'layers.0.enc_self_attn.W_K.bias', 'layers.0.enc_self_attn.W_V.weight', 'layers.0.enc_self_attn.W_V.bias', 'layers.0.pos_ffn.fc.0.weight', 'layers.0.pos_ffn.fc.2.weight', 'layers.0.pos_ffn.layernorm.weight', 'layers.0.pos_ffn.layernorm.bias', 'layers.1.enc_self_attn.linear.weight', 'layers.1.enc_self_attn.linear.bias', 'layers.1.enc_self_attn.layernorm.weight', 'layers.1.enc_self_attn.layernorm.bias', 'layers.1.enc_self_attn.W_Q.weight', 'layers.1.enc_self_attn.W_Q.bias', 'layers.1.enc_self_attn.W_K.weight', 'layers.1.enc_self_attn.W_K.bias', 'layers.1.enc_self_attn.W_V.weight', 'layers.1.enc_self_attn.W_V.bias', 'layers.1.pos_ffn.fc.0.weight', 'layers.1.pos_ffn.fc.2.weight', 'layers.1.pos_ffn.layernorm.weight', 'layers.1.pos_ffn.layernorm.bias', 'layers.2.enc_self_attn.linear.weight', 'layers.2.enc_self_attn.linear.bias', 'layers.2.enc_self_attn.layernorm.weight', 'layers.2.enc_self_attn.layernorm.bias', 'layers.2.enc_self_attn.W_Q.weight', 'layers.2.enc_self_attn.W_Q.bias', 'layers.2.enc_self_attn.W_K.weight', 'layers.2.enc_self_attn.W_K.bias', 'layers.2.enc_self_attn.W_V.weight', 'layers.2.enc_self_attn.W_V.bias', 'layers.2.pos_ffn.fc.0.weight', 'layers.2.pos_ffn.fc.2.weight', 'layers.2.pos_ffn.layernorm.weight', 'layers.2.pos_ffn.layernorm.bias', 'layers.3.enc_self_attn.linear.weight', 'layers.3.enc_self_attn.linear.bias', 'layers.3.enc_self_attn.layernorm.weight', 'layers.3.enc_self_attn.layernorm.bias', 'layers.3.enc_self_attn.W_Q.weight', 'layers.3.enc_self_attn.W_Q.bias', 'layers.3.enc_self_attn.W_K.weight', 'layers.3.enc_self_attn.W_K.bias', 'layers.3.enc_self_attn.W_V.weight', 'layers.3.enc_self_attn.W_V.bias', 'layers.3.pos_ffn.fc.0.weight', 'layers.3.pos_ffn.fc.2.weight', 'layers.3.pos_ffn.layernorm.weight', 'layers.3.pos_ffn.layernorm.bias', 'layers.4.enc_self_attn.linear.weight', 'layers.4.enc_self_attn.linear.bias', 'layers.4.enc_self_attn.layernorm.weight', 'layers.4.enc_self_attn.layernorm.bias', 'layers.4.enc_self_attn.W_Q.weight', 'layers.4.enc_self_attn.W_Q.bias', 'layers.4.enc_self_attn.W_K.weight', 'layers.4.enc_self_attn.W_K.bias', 'layers.4.enc_self_attn.W_V.weight', 'layers.4.enc_self_attn.W_V.bias', 'layers.4.pos_ffn.fc.0.weight', 'layers.4.pos_ffn.fc.2.weight', 'layers.4.pos_ffn.layernorm.weight', 'layers.4.pos_ffn.layernorm.bias', 'layers.5.enc_self_attn.linear.weight', 'layers.5.enc_self_attn.linear.bias', 'layers.5.enc_self_attn.layernorm.weight', 'layers.5.enc_self_attn.layernorm.bias', 'layers.5.enc_self_attn.W_Q.weight', 'layers.5.enc_self_attn.W_Q.bias', 'layers.5.enc_self_attn.W_K.weight', 'layers.5.enc_self_attn.W_K.bias', 'layers.5.enc_self_attn.W_V.weight', 'layers.5.enc_self_attn.W_V.bias', 'layers.5.pos_ffn.fc.0.weight', 'layers.5.pos_ffn.fc.2.weight', 'layers.5.pos_ffn.layernorm.weight', 'layers.5.pos_ffn.layernorm.bias', 'fc.1.weight', 'fc.1.bias', 'fc.3.weight', 'fc.3.bias', 'classifier_global.weight', 'classifier_global.bias', 'classifier_atom.weight', 'classifier_atom.bias']
        elif self.pretrain_layer == 'all_12layer':
            pretrained_parameters = ['embedding.tok_embed.weight', 'embedding.pos_embed.weight',
                                     'embedding.norm.weight', 'embedding.norm.bias',
                                     'layers.0.enc_self_attn.linear.weight', 'layers.0.enc_self_attn.linear.bias',
                                     'layers.0.enc_self_attn.layernorm.weight', 'layers.0.enc_self_attn.layernorm.bias',
                                     'layers.0.enc_self_attn.W_Q.weight', 'layers.0.enc_self_attn.W_Q.bias',
                                     'layers.0.enc_self_attn.W_K.weight', 'layers.0.enc_self_attn.W_K.bias',
                                     'layers.0.enc_self_attn.W_V.weight', 'layers.0.enc_self_attn.W_V.bias',
                                     'layers.0.pos_ffn.fc.0.weight', 'layers.0.pos_ffn.fc.2.weight',
                                     'layers.0.pos_ffn.layernorm.weight', 'layers.0.pos_ffn.layernorm.bias',
                                     'layers.1.enc_self_attn.linear.weight', 'layers.1.enc_self_attn.linear.bias',
                                     'layers.1.enc_self_attn.layernorm.weight', 'layers.1.enc_self_attn.layernorm.bias',
                                     'layers.1.enc_self_attn.W_Q.weight', 'layers.1.enc_self_attn.W_Q.bias',
                                     'layers.1.enc_self_attn.W_K.weight', 'layers.1.enc_self_attn.W_K.bias',
                                     'layers.1.enc_self_attn.W_V.weight', 'layers.1.enc_self_attn.W_V.bias',
                                     'layers.1.pos_ffn.fc.0.weight', 'layers.1.pos_ffn.fc.2.weight',
                                     'layers.1.pos_ffn.layernorm.weight', 'layers.1.pos_ffn.layernorm.bias',
                                     'layers.2.enc_self_attn.linear.weight', 'layers.2.enc_self_attn.linear.bias',
                                     'layers.2.enc_self_attn.layernorm.weight', 'layers.2.enc_self_attn.layernorm.bias',
                                     'layers.2.enc_self_attn.W_Q.weight', 'layers.2.enc_self_attn.W_Q.bias',
                                     'layers.2.enc_self_attn.W_K.weight', 'layers.2.enc_self_attn.W_K.bias',
                                     'layers.2.enc_self_attn.W_V.weight', 'layers.2.enc_self_attn.W_V.bias',
                                     'layers.2.pos_ffn.fc.0.weight', 'layers.2.pos_ffn.fc.2.weight',
                                     'layers.2.pos_ffn.layernorm.weight', 'layers.2.pos_ffn.layernorm.bias',
                                     'layers.3.enc_self_attn.linear.weight', 'layers.3.enc_self_attn.linear.bias',
                                     'layers.3.enc_self_attn.layernorm.weight', 'layers.3.enc_self_attn.layernorm.bias',
                                     'layers.3.enc_self_attn.W_Q.weight', 'layers.3.enc_self_attn.W_Q.bias',
                                     'layers.3.enc_self_attn.W_K.weight', 'layers.3.enc_self_attn.W_K.bias',
                                     'layers.3.enc_self_attn.W_V.weight', 'layers.3.enc_self_attn.W_V.bias',
                                     'layers.3.pos_ffn.fc.0.weight', 'layers.3.pos_ffn.fc.2.weight',
                                     'layers.3.pos_ffn.layernorm.weight', 'layers.3.pos_ffn.layernorm.bias',
                                     'layers.4.enc_self_attn.linear.weight', 'layers.4.enc_self_attn.linear.bias',
                                     'layers.4.enc_self_attn.layernorm.weight', 'layers.4.enc_self_attn.layernorm.bias',
                                     'layers.4.enc_self_attn.W_Q.weight', 'layers.4.enc_self_attn.W_Q.bias',
                                     'layers.4.enc_self_attn.W_K.weight', 'layers.4.enc_self_attn.W_K.bias',
                                     'layers.4.enc_self_attn.W_V.weight', 'layers.4.enc_self_attn.W_V.bias',
                                     'layers.4.pos_ffn.fc.0.weight', 'layers.4.pos_ffn.fc.2.weight',
                                     'layers.4.pos_ffn.layernorm.weight', 'layers.4.pos_ffn.layernorm.bias',
                                     'layers.5.enc_self_attn.linear.weight', 'layers.5.enc_self_attn.linear.bias',
                                     'layers.5.enc_self_attn.layernorm.weight', 'layers.5.enc_self_attn.layernorm.bias',
                                     'layers.5.enc_self_attn.W_Q.weight', 'layers.5.enc_self_attn.W_Q.bias',
                                     'layers.5.enc_self_attn.W_K.weight', 'layers.5.enc_self_attn.W_K.bias',
                                     'layers.5.enc_self_attn.W_V.weight', 'layers.5.enc_self_attn.W_V.bias',
                                     'layers.5.pos_ffn.fc.0.weight', 'layers.5.pos_ffn.fc.2.weight',
                                     'layers.5.pos_ffn.layernorm.weight', 'layers.5.pos_ffn.layernorm.bias',

                                     'layers.6.enc_self_attn.linear.weight', 'layers.6.enc_self_attn.linear.bias',
                                     'layers.6.enc_self_attn.layernorm.weight', 'layers.6.enc_self_attn.layernorm.bias',
                                     'layers.6.enc_self_attn.W_Q.weight', 'layers.6.enc_self_attn.W_Q.bias',
                                     'layers.6.enc_self_attn.W_K.weight', 'layers.6.enc_self_attn.W_K.bias',
                                     'layers.6.enc_self_attn.W_V.weight', 'layers.6.enc_self_attn.W_V.bias',
                                     'layers.6.pos_ffn.fc.0.weight', 'layers.6.pos_ffn.fc.2.weight',
                                     'layers.6.pos_ffn.layernorm.weight', 'layers.6.pos_ffn.layernorm.bias',
                                     'layers.7.enc_self_attn.linear.weight', 'layers.7.enc_self_attn.linear.bias',
                                     'layers.7.enc_self_attn.layernorm.weight', 'layers.7.enc_self_attn.layernorm.bias',
                                     'layers.7.enc_self_attn.W_Q.weight', 'layers.7.enc_self_attn.W_Q.bias',
                                     'layers.7.enc_self_attn.W_K.weight', 'layers.7.enc_self_attn.W_K.bias',
                                     'layers.7.enc_self_attn.W_V.weight', 'layers.7.enc_self_attn.W_V.bias',
                                     'layers.7.pos_ffn.fc.0.weight', 'layers.7.pos_ffn.fc.2.weight',
                                     'layers.7.pos_ffn.layernorm.weight', 'layers.7.pos_ffn.layernorm.bias',
                                     'layers.8.enc_self_attn.linear.weight', 'layers.8.enc_self_attn.linear.bias',
                                     'layers.8.enc_self_attn.layernorm.weight', 'layers.8.enc_self_attn.layernorm.bias',
                                     'layers.8.enc_self_attn.W_Q.weight', 'layers.8.enc_self_attn.W_Q.bias',
                                     'layers.8.enc_self_attn.W_K.weight', 'layers.8.enc_self_attn.W_K.bias',
                                     'layers.8.enc_self_attn.W_V.weight', 'layers.8.enc_self_attn.W_V.bias',
                                     'layers.8.pos_ffn.fc.0.weight', 'layers.8.pos_ffn.fc.2.weight',
                                     'layers.8.pos_ffn.layernorm.weight', 'layers.8.pos_ffn.layernorm.bias',
                                     'layers.9.enc_self_attn.linear.weight', 'layers.9.enc_self_attn.linear.bias',
                                     'layers.9.enc_self_attn.layernorm.weight', 'layers.9.enc_self_attn.layernorm.bias',
                                     'layers.9.enc_self_attn.W_Q.weight', 'layers.9.enc_self_attn.W_Q.bias',
                                     'layers.9.enc_self_attn.W_K.weight', 'layers.9.enc_self_attn.W_K.bias',
                                     'layers.9.enc_self_attn.W_V.weight', 'layers.9.enc_self_attn.W_V.bias',
                                     'layers.9.pos_ffn.fc.0.weight', 'layers.9.pos_ffn.fc.2.weight',
                                     'layers.9.pos_ffn.layernorm.weight', 'layers.9.pos_ffn.layernorm.bias',
                                     'layers.10.enc_self_attn.linear.weight', 'layers.10.enc_self_attn.linear.bias',
                                     'layers.10.enc_self_attn.layernorm.weight',
                                     'layers.10.enc_self_attn.layernorm.bias', 'layers.10.enc_self_attn.W_Q.weight',
                                     'layers.10.enc_self_attn.W_Q.bias', 'layers.10.enc_self_attn.W_K.weight',
                                     'layers.10.enc_self_attn.W_K.bias', 'layers.10.enc_self_attn.W_V.weight',
                                     'layers.10.enc_self_attn.W_V.bias', 'layers.10.pos_ffn.fc.0.weight',
                                     'layers.10.pos_ffn.fc.2.weight', 'layers.10.pos_ffn.layernorm.weight',
                                     'layers.10.pos_ffn.layernorm.bias'
                                     'fc.1.weight', 'fc.1.bias', 'fc.3.weight', 'fc.3.bias', 'classifier_global.weight',
                                     'classifier_global.bias', 'classifier_atom.weight', 'classifier_atom.bias']

        pretrained_model = torch.load(self.pretrained_model, map_location=torch.device('cpu'))
        # pretrained_model = torch.load(self.pretrained_model)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_model['model_state_dict'].items() if k in pretrained_parameters}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)

    def load_pretrained_model_continue(self, model):
        pretrained_parameters = ['embedding.tok_embed.weight', 'embedding.pos_embed.weight', 'embedding.norm.weight',
                                 'embedding.norm.bias', 'layers.0.enc_self_attn.linear.weight',
                                 'layers.0.enc_self_attn.linear.bias', 'layers.0.enc_self_attn.layernorm.weight',
                                 'layers.0.enc_self_attn.layernorm.bias', 'layers.0.enc_self_attn.W_Q.weight',
                                 'layers.0.enc_self_attn.W_Q.bias', 'layers.0.enc_self_attn.W_K.weight',
                                 'layers.0.enc_self_attn.W_K.bias', 'layers.0.enc_self_attn.W_V.weight',
                                 'layers.0.enc_self_attn.W_V.bias', 'layers.0.pos_ffn.fc1.weight',
                                 'layers.0.pos_ffn.fc1.bias', 'layers.0.pos_ffn.fc2.weight',
                                 'layers.0.pos_ffn.fc2.bias', 'layers.1.enc_self_attn.linear.weight',
                                 'layers.1.enc_self_attn.linear.bias', 'layers.1.enc_self_attn.layernorm.weight',
                                 'layers.1.enc_self_attn.layernorm.bias', 'layers.1.enc_self_attn.W_Q.weight',
                                 'layers.1.enc_self_attn.W_Q.bias', 'layers.1.enc_self_attn.W_K.weight',
                                 'layers.1.enc_self_attn.W_K.bias', 'layers.1.enc_self_attn.W_V.weight',
                                 'layers.1.enc_self_attn.W_V.bias', 'layers.1.pos_ffn.fc1.weight',
                                 'layers.1.pos_ffn.fc1.bias', 'layers.1.pos_ffn.fc2.weight',
                                 'layers.1.pos_ffn.fc2.bias', 'layers.2.enc_self_attn.linear.weight',
                                 'layers.2.enc_self_attn.linear.bias', 'layers.2.enc_self_attn.layernorm.weight',
                                 'layers.2.enc_self_attn.layernorm.bias', 'layers.2.enc_self_attn.W_Q.weight',
                                 'layers.2.enc_self_attn.W_Q.bias', 'layers.2.enc_self_attn.W_K.weight',
                                 'layers.2.enc_self_attn.W_K.bias', 'layers.2.enc_self_attn.W_V.weight',
                                 'layers.2.enc_self_attn.W_V.bias', 'layers.2.pos_ffn.fc1.weight',
                                 'layers.2.pos_ffn.fc1.bias', 'layers.2.pos_ffn.fc2.weight',
                                 'layers.2.pos_ffn.fc2.bias', 'layers.3.enc_self_attn.linear.weight',
                                 'layers.3.enc_self_attn.linear.bias', 'layers.3.enc_self_attn.layernorm.weight',
                                 'layers.3.enc_self_attn.layernorm.bias', 'layers.3.enc_self_attn.W_Q.weight',
                                 'layers.3.enc_self_attn.W_Q.bias', 'layers.3.enc_self_attn.W_K.weight',
                                 'layers.3.enc_self_attn.W_K.bias', 'layers.3.enc_self_attn.W_V.weight',
                                 'layers.3.enc_self_attn.W_V.bias', 'layers.3.pos_ffn.fc1.weight',
                                 'layers.3.pos_ffn.fc1.bias', 'layers.3.pos_ffn.fc2.weight',
                                 'layers.3.pos_ffn.fc2.bias', 'layers.4.enc_self_attn.linear.weight',
                                 'layers.4.enc_self_attn.linear.bias', 'layers.4.enc_self_attn.layernorm.weight',
                                 'layers.4.enc_self_attn.layernorm.bias', 'layers.4.enc_self_attn.W_Q.weight',
                                 'layers.4.enc_self_attn.W_Q.bias', 'layers.4.enc_self_attn.W_K.weight',
                                 'layers.4.enc_self_attn.W_K.bias', 'layers.4.enc_self_attn.W_V.weight',
                                 'layers.4.enc_self_attn.W_V.bias', 'layers.4.pos_ffn.fc1.weight',
                                 'layers.4.pos_ffn.fc1.bias', 'layers.4.pos_ffn.fc2.weight',
                                 'layers.4.pos_ffn.fc2.bias', 'layers.5.enc_self_attn.linear.weight',
                                 'layers.5.enc_self_attn.linear.bias', 'layers.5.enc_self_attn.layernorm.weight',
                                 'layers.5.enc_self_attn.layernorm.bias', 'layers.5.enc_self_attn.W_Q.weight',
                                 'layers.5.enc_self_attn.W_Q.bias', 'layers.5.enc_self_attn.W_K.weight',
                                 'layers.5.enc_self_attn.W_K.bias', 'layers.5.enc_self_attn.W_V.weight',
                                 'layers.5.enc_self_attn.W_V.bias', 'layers.5.pos_ffn.fc1.weight',
                                 'layers.5.pos_ffn.fc1.bias', 'layers.5.pos_ffn.fc2.weight',
                                 'layers.5.pos_ffn.fc2.bias', 'fc.1.weight', 'fc.1.bias', 'fc.3.weight', 'fc.3.bias',
                                 'fc.5.weight', 'fc.5.bias', 'fc.7.weight', 'fc.7.bias', 'classifier_global.weight',
                                 'classifier_global.bias', 'classifier_atom.weight', 'classifier_atom.bias']
        pretrained_model = torch.load(self.pretrained_model, map_location=torch.device('cpu'))
        # pretrained_model = torch.load(self.pretrained_model)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_model['model_state_dict'].items() if k in pretrained_parameters}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)



