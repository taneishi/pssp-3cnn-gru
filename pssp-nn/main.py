import numpy as np
import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
import torch
import timeit
import argparse
import os

N_STATE = 8
N_AA = 21

class CrossEntropy(object):
    def __init__(self):
        pass

    def __call__(self, out, target, seq_len):
        loss = 0
        for o, t, l in zip(out, target, seq_len):
            loss += nn.CrossEntropyLoss()(o[:l], t[:l])
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

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, seq_len):
        self.X = X
        self.y = y.astype(int)
        self.seq_len = seq_len.astype(int)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.seq_len[idx]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        conv_hidden_size = 64

        self.conv1 = nn.Sequential(
            nn.Conv1d(N_AA, conv_hidden_size, 3, 1, 3 // 2),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv1d(N_AA, conv_hidden_size, 7, 1, 7 // 2),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv1d(N_AA, conv_hidden_size, 11, 1, 11 // 2),
            nn.ReLU())

        # LSTM(input_size, hidden_size, num_layers, bias,
        #      batch_first, dropout, bidirectional)
        rnn_hidden_size = 256
        self.brnn = nn.GRU(conv_hidden_size*3, rnn_hidden_size, 3, True, True, 0.5, True)

        self.fc = nn.Sequential(
                nn.Linear(rnn_hidden_size*2+conv_hidden_size*3, 128),
                nn.ReLU(),
                nn.Linear(128, N_STATE),
                nn.ReLU())

    def forward(self, x):
        # obtain multiple local contextual feature map
        conv_out = torch.cat([self.conv1(x), self.conv2(x), self.conv3(x)], dim=1)

        # Turn (batch_size x hidden_size x seq_len)
        # into (batch_size x seq_len x hidden_size)
        conv_out = conv_out.transpose(1, 2)

        # bidirectional rnn
        out, _ = self.brnn(conv_out)

        out = torch.cat([conv_out, out], dim=2)
        # print(out.sum())

        # Output shape is (batch_size x seq_len x classnum)
        out = self.fc(out)
        out = F.softmax(out, dim=2)
        return out

def train(model, device, epoch, train_loader, optimizer, loss_function):
    model.train()
    train_loss = 0
    for index, (data, target, seq_len) in enumerate(train_loader, 1):
        data, target, seq_len = data.to(device), target.to(device), seq_len.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_function(out, target, seq_len)
        train_loss += loss.item() / len(data)
        acc = accuracy(out, target, seq_len)
        loss.backward()
        optimizer.step()
        print('\repoch%3d [%3d/%3d] train loss %6.4f train acc%6.3f' % (epoch, index, len(train_loader), train_loss / index, acc), end='')

def test(model, device, test_loader, loss_function):
    model.eval()
    test_loss = 0
    acc = 0
    with torch.no_grad():
        for data, target, seq_len in test_loader:
            data, target, seq_len = data.to(device), target.to(device), seq_len.to(device)
            out = model(data)
            loss = loss_function(out, target, seq_len)
            test_loss += loss.item()
            acc += accuracy(out, target, seq_len)

    test_loss /= len(test_loader.dataset)
    acc /= len(test_loader)
    return test_loss, acc

def main():
    # params
    # ----------
    parser = argparse.ArgumentParser(description='Protein Secondary Structure Prediction')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='The number of epochs to run (default: 100)')
    parser.add_argument('-b', '--batch_size_train', type=int, default=100,
                        help='input batch size for training (default: 100)')
    parser.add_argument('-b_test', '--batch_size_test', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--result_dir', type=str, default='./result',
                        help='Output directory (default: ./result)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using %s device.' % device)

    # make directory to save trained model
    os.makedirs(args.result_dir, exist_ok=True)

    # laod dataset and set k-fold cross validation
    X_train, y_train, seq_len_train, X_test, y_test, seq_len_test = np.load('dataset.npz').values()

    train_loader = torch.utils.data.DataLoader(MyDataset(X_train, y_train, seq_len_train), batch_size=args.batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(MyDataset(X_test, y_test, seq_len_test), batch_size=args.batch_size_test, shuffle=False)

    print('train %d test %d' % (len(train_loader.dataset), len(test_loader.dataset)))

    # model, loss_function, optimizer
    model = Net().to(device)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    loss_function = CrossEntropy()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.01)

    start = timeit.default_timer()

    # train and test
    for epoch in range(1, args.epochs+1):
        epoch_start = timeit.default_timer()
        train(model, device, epoch, train_loader, optimizer, loss_function)
        test_loss, acc = test(model, device, test_loader, loss_function)
        print(' test_loss %6.4f acc%6.3f time %5.1fs' % (test_loss, acc, timeit.default_timer() - epoch_start))

    # save trained model
    torch.save(model.state_dict(), os.path.join(args.result_dir, 'model.pth'))

if __name__ == '__main__':
    torch.manual_seed(123)
    main()
