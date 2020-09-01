import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import argparse
import timeit

from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
import transformer.Constants as Constants
from dataset import TranslationDataset, paired_collate_fn

def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct

def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum() # average later
    else:
        # print(gold)
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss

def prepare_dataloaders(data, opt):
    # ========= Preparing DataLoader =========#
    train_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['train']['src'],
            tgt_insts=data['train']['tgt']),
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['valid']['src'],
            tgt_insts=data['valid']['tgt']),
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)
    return train_loader, valid_loader

def train(net, training_data, optimizer, device, smoothing):
    ''' Epoch operation in training phase'''
    net.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    for batch in training_data:
        # prepare data
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)

        gold = tgt_seq[:, 1:]

        # forward
        optimizer.zero_grad()
        pred = net(src_seq, src_pos, tgt_seq, tgt_pos)

        # backward
        loss, n_correct = cal_performance(pred, gold, smoothing=smoothing)
        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy

def test(net, test_data, device):
    ''' Epoch operation in evaluation phase '''
    net.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in test_data:
            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]

            # forward
            pred = net(src_seq, src_pos, tgt_seq, tgt_pos)
            loss, n_correct = cal_performance(pred, gold, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy

def main(args):
    ''' Main function '''
    #========= Loading Dataset =========#
    data = torch.load(args.data)
    args.max_token_seq_len = data['settings'].max_token_seq_len

    training_data, test_data = prepare_dataloaders(data, args)

    args.src_vocab_size = training_data.dataset.src_vocab_size
    args.tgt_vocab_size = training_data.dataset.tgt_vocab_size

    #========= Preparing Model =========#
    if args.embs_share_weight:
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
            'The src/tgt word2idx table are different but asked to share word embedding.'

    device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')
    print('Using %s device.' % device)

    net = Transformer(
        args.src_vocab_size,
        args.tgt_vocab_size,
        args.max_token_seq_len,
        tgt_emb_prj_weight_sharing=args.proj_share_weight,
        emb_src_tgt_weight_sharing=args.embs_share_weight,
        d_k=args.d_k,
        d_v=args.d_v,
        d_model=args.d_model,
        d_word_vec=args.d_word_vec,
        d_inner=args.d_inner_hid,
        n_layers=args.n_layers,
        n_head=args.n_head,
        dropout=args.dropout).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, net.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        args.d_model, args.n_warmup_steps)

    test_losses = []
    for epoch in range(args.epoch+1):
        epoch_start = timeit.default_timer()

        train_loss, train_acc = train(net, training_data, optimizer, device, smoothing=args.label_smoothing)
        test_loss, test_acc = test(net, test_data, device)

        print('[%03d/%03d] train_loss %6.3f test_loss: %6.3f time %5.2fsec' % (
            epoch, args.epoch, train_loss, test_loss, (timeit.default_timer() - epoch_start)))

        test_losses.append(test_loss)

        if min(test_losses) >= test_loss:
            save_path = '%s/%5.3f.pth' % (args.model_dir, test_loss)
            torch.save(net.state_dict(), save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/dataset.pt')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--d_word_vec', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--d_inner_hid', type=int, default=512)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_warmup_steps', type=int, default=4000)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--embs_share_weight', action='store_true')
    parser.add_argument('--proj_share_weight', action='store_true')
    parser.add_argument('--model_dir', type=str, default='model')
    parser.add_argument('--label_smoothing', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    print(vars(args))

    main(args)
