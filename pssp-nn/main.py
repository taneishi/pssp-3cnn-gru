import numpy as np
import torch.utils.data
import torch.nn as nn
import torch
import timeit
import argparse
import os

from model import Net

class CrossEntropy(object):
    def __init__(self):
        self.loss_function = nn.CrossEntropyLoss()

    def __call__(self, out, target, seq_len):
        loss = sum(self.loss_function(o[:l], t[:l]) for o, t, l in zip(out, target, seq_len))
        return loss

def accuracy(out, target, seq_len):
    '''
    out.shape : (batch_size, seq_len, class_num)
    target.shape : (class_num, seq_len)
    '''
    out = out.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    seq_len = seq_len.cpu().detach().numpy()
    out = out.argmax(axis=2)

    return np.array([np.equal(o[:l], t[:l]).sum()/l for o, t, l in zip(out, target, seq_len)]).mean()

def train(train_loader, model, optimizer, loss_function, epoch):
    train_loss = 0
    model.train()

    for index, (data, target, seq_len) in enumerate(train_loader, 1):
        optimizer.zero_grad()
        out = model(data)
        loss = loss_function(out, target, seq_len)
        train_loss += loss.item() / len(data)
        acc = accuracy(out, target, seq_len)
        loss.backward()
        optimizer.step()

        print('\repoch%3d [%3d/%3d] train_loss %5.3f train_acc %5.3f' % (epoch, index, len(train_loader), train_loss / index, acc), end='')

def test(test_loader, model, loss_function):
    test_loss = 0
    acc = 0
    model.eval()

    for index, (data, target, seq_len) in enumerate(test_loader, 1):
        with torch.no_grad():
            out = model(data)
        loss = loss_function(out, target, seq_len)
        test_loss += loss.item() / len(data)
        acc += accuracy(out, target, seq_len)
    
    print(' test_loss %5.3f test_acc %5.3f' % (test_loss / index, acc / index), end='')

    return test_loss, acc

def main():
    # params
    parser = argparse.ArgumentParser(description='Protein Secondary Structure Prediction')
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-b', '--batch_size_train', type=int, default=100,
                        help='input batch size for training (default: 100)')
    parser.add_argument('-b_test', '--batch_size_test', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using %s device.' % device)

    # laod dataset 
    X_train, y_train, seq_len_train, X_test, y_test, seq_len_test = np.load('../pssp-data/dataset.npz').values()

    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    seq_len_train = torch.ShortTensor(seq_len_train).to(device)

    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.LongTensor(y_test).to(device)
    seq_len_test = torch.ShortTensor(seq_len_test).to(device)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train, seq_len_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test, seq_len_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_test, shuffle=False)

    print('train %d test %d' % (len(train_loader.dataset), len(test_loader.dataset)))

    # model, loss_function, optimizer
    model = Net().to(device)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    loss_function = CrossEntropy()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.01)

    for epoch in range(args.epochs):
        epoch_start = timeit.default_timer()

        train(train_loader, model, optimizer, loss_function, epoch)
        test(test_loader, model, loss_function)

        print(' %5.1f sec' % (timeit.default_timer() - epoch_start))

    # save trained model
    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    torch.manual_seed(123)
    main()
