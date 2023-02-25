'''
load trained bmtl model
1. tsne plot
2. fast adaptation test
'''
import collections
from torch.optim import lr_scheduler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sn
from bmtl_model import MultiTaskModel
import torch
from torch.utils.data import Dataset, DataLoader
from bmtl_loss import MultiTaskLossWrapper
import torch.nn as nn
import sklearn.metrics as metrics
import argparse
from sklearn.manifold import TSNE
from tools import balanced_sample_maker

parser = argparse.ArgumentParser(description='PyTorch USCHAD Training')
parser.add_argument('--batch_size', default=10, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.03, type=float, help='initial learning rate')
parser.add_argument('--noise_mode', default='asym')
parser.add_argument('--r', default=0.4, type=float, help='noise ratio')
parser.add_argument('--leave_sub', default=14, type=int, help='leave sub id')
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--num_samples', default=1, type=int)
parser.add_argument('--num_workers', default=0, type=int)
args = parser.parse_args()

# Hardcoded length of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_LENGTH = 200

NUM_CHANNEL = 6

device = 'cuda'

HIDDEN_UNIT = 128

NUM_SUBS = 13

PATH = './models/leave' + str(args.leave_sub) + '/' + str(args.r) + '_' + args.noise_mode + '/'
print(PATH)

# =============================================================================
# load data and trained model
# =============================================
def load_tensor(name):
    # Read the array from disk
    new_data = np.loadtxt('E:/2022_spring/lnl_iflf/data/processed_uschad/Sub' + str(name) + '_data.txt')

    # Note that this returned a 2D array!
    print(new_data.shape)

    # However, going back to 3D is easy if we know the
    # original shape of the array
    new_data = new_data.reshape((-1, SLIDING_WINDOW_LENGTH, NUM_CHANNEL))
    return new_data


# load model
def load_pretrain():
    model_path = PATH + '/bmtl_model_leave' + str(args.leave_sub) + '_' + str(args.r) + '_' + args.noise_mode
    model = MultiTaskModel()
    model.load_state_dict(torch.load(model_path))
    return model


def load_data_by_id(subid):
    X = load_tensor(subid)
    y = np.loadtxt("E:/2022_spring/lnl_iflf/data/processed_uschad/Sub" + str(subid) + "_label.txt", dtype=int)
    y = y - 1
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.8,
                                                        random_state=42,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test


# =============================================================================
# prepare for fast adaptation test
# =============================================================================

labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
lb = preprocessing.LabelBinarizer()
lb.fit(y=labels)


# =============================================================================
# do t-sne of test data
# =============================================================================
def tsne(test, ytest):
    data1 = np.hstack((test, ytest.reshape(-1, 1)))
    Y = TSNE(n_components=2).fit_transform(data1)

    fig, ax = plt.subplots()
    groups = pd.DataFrame(Y, columns=['x', 'y']).assign(category=ytest).groupby('category')
    listact = ['walkF', 'walkL', 'walkR', 'upstairs', 'downstairs', 'run',
               'jump', 'sit', 'stand', 'lying']
    # listact = ['walk','jog','stairs','sit','stand','soccer','basketball']
    # colors = cm.rainbow(np.linspace(0, 1, len(listact)))
    ind = 0
    for name, points in groups:
        f = listact[int(name - 1)]
        print(f)
        # ax.scatter(points.x, points.y, label=f, color=colors[ind])
        ax.scatter(points.x, points.y, label=f)
        ind += 1
    ax.legend()
    fig.savefig(PATH + str(args.num_samples) + '_samples_tsne.png')



##################################################################################
# still have to define a dataset and train, val, test dataloader first
################################################################################

class uschad_dataset(Dataset):
    def __init__(self, mode, leave_sub):
        self.mode = mode
        # load data for test
        xtrain, X_test, ytrain, y_test = load_data_by_id(leave_sub)
        X_train, y_train, xval, yval = balanced_sample_maker(xtrain, ytrain, args.num_samples)

        if self.mode == 'train':
            self.data = X_train
            self.label = y_train
            print('data for fast adaptation')
            print(collections.Counter(y_train))

        elif self.mode == 'val':
            self.data = xval
            self.label = yval
        elif self.mode == 'test':
            self.data = X_test
            self.label = y_test
        else:  # get all data for tsne
            self.data = np.vstack((X_train, xval, X_test))
            self.label = np.concatenate((y_train, yval, y_test))

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.label)


class uschad_dataloader():  # dataloader for feature extractor training
    def __init__(self, batch_size, leave_sub, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = uschad_dataset('train', leave_sub)
        self.eval_dataset = uschad_dataset('val', leave_sub)
        self.test_dataset = uschad_dataset('test', leave_sub)
        self.all_dataset = uschad_dataset('all', leave_sub)

    def run(self, mode):
        if mode == 'train':
            train_loader = DataLoader(dataset=self.train_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=self.num_workers,
                                      drop_last=True)
            return train_loader

        elif mode == 'val':
            val_loader = DataLoader(dataset=self.eval_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=self.num_workers,
                                    drop_last=True)
            return val_loader
        elif mode == 'test':
            test_loader = DataLoader(dataset=self.test_dataset,
                                     batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     drop_last=True)
            return test_loader
        else:
            all_loader = DataLoader(dataset=self.all_dataset,
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers,
                                    drop_last=True)
            return all_loader


def make_inputs(inputs, labels):

    res, targets = [], []
    for i in range(13):
        # make target onehot vec
        targets.append(labels[i].long().to(device))
        res.append(inputs[i].to(device))
    return res, targets


###############################################################################################
# the eval procedure (train with small amount of clean data and early stopping, then test)
###########################################################################################

dl = uschad_dataloader(batch_size=args.batch_size, leave_sub=args.leave_sub, num_workers=args.num_workers)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def plot_tsne(all_loader):
    model = load_pretrain().cuda()  # send model to gpu
    m_encoder = model.encoder

    m_encoder.dropout = Identity()  # discard the last dropout layer
    # initialize hidden state
    h = m_encoder.init_hidden(args.batch_size)

    outputs, ys = [], []
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(all_loader):
            inputs, labels = inputs.cuda(), labels.cuda()

            output, h = m_encoder(inputs.float(), h, args.batch_size)
            output = output.view(args.batch_size, -1, 128)[:, -1, :]
            outputs.append(output.detach().cpu().numpy())

            ys.append(labels.detach().cpu().numpy())

    outputs = np.concatenate(outputs)
    ys = np.concatenate(ys)
    tsne(outputs, ys)


all_loader = dl.run('all')
plot_tsne(all_loader)

# initialize global variable for ELR loss (need NUM_SUBS)
NUM_EXAMP = len(all_loader.dataset)
loss_func = MultiTaskLossWrapper(NUM_EXAMP, args.num_class, NUM_SUBS).to(
    device)  # just making sure the loss is on the gpu


##########################################################################################
# finetune the model, early stopping with val
##########################################################################################

def get_accuracy(y_prob, y_true):
    _, preds = torch.max(y_prob, 1)
    assert y_true.size() == preds.size()
    return preds.eq(y_true).cpu().sum().item() / y_true.size(0)


# define the model class
class AdaptModel(nn.Module):
    def __init__(self):
        super(AdaptModel, self).__init__()
        model = load_pretrain()
        self.encoder = model.encoder
        self.fc = nn.Linear(HIDDEN_UNIT,
                            args.num_class)

    def forward(self, x, hidden, batch_size):
        x0, _ = self.encoder(x.float(), hidden, batch_size)
        x0 = self.fc(x0)
        t = x0.view(batch_size, -1, args.num_class)[:, -1, :]
        return t


def val(net, val_dl, batch_size):
    val_h = net.encoder.init_hidden(batch_size)
    val_losses = []
    accuracy = 0
    f1score = 0
    net.eval()

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_dl):
            inputs, targets = inputs.cuda(), labels.cuda()

            val_h = tuple([each.data for each in val_h])

            output = net(inputs.float(), val_h, batch_size)

            val_loss = CEloss(output, targets.long())
            val_losses.append(val_loss.item())

            top_p, top_class = output.topk(1, dim=1)
            equals = top_class == targets.view(*top_class.shape).long()
            accuracy += torch.mean(equals.type(torch.FloatTensor))
            f1score += metrics.f1_score(top_class.cpu(), targets.view(*top_class.shape).long().cpu(),
                                        average='weighted')
    return val_losses, accuracy / len(val_dl), f1score / len(val_dl)


def train(net, train_dl, val_dl, epochs=50, batch_size=64, lr=0.03, delta=0.001, patience=5):
    opt = torch.optim.SGD(net.fc.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(opt, step_size=100, gamma=0.1)  # decay every 100 epochs

    last_loss = 100
    trigger_times = 0
    net.train()  # train the linear layer only

    net.encoder.eval()  # make the encoder not trainable
    for param in net.encoder.parameters():
        param.requires_grad = False

    res = []
    for e in range(epochs):
        train_acc = 0
        # initialize hidden state
        h = net.encoder.init_hidden(batch_size)
        train_losses = []

        for batch_idx, (inputs, labels) in enumerate(train_dl):
            inputs, targets = inputs.cuda(), labels.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            opt.zero_grad()

            # get the output from the model
            output = net(inputs.float(), h, batch_size)

            loss = CEloss(output, targets.long())
            train_losses.append(loss.item())
            train_acc += get_accuracy(output, targets)
            loss.backward()
            opt.step()

        # Early stopping
        val_losses, accuracy, f1score = val(net, val_dl, batch_size)

        if np.mean(val_losses) + delta >= last_loss:  # allow very small difference
            trigger_times += 1
            print('Trigger Times:', trigger_times)

            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                print('model trained and saved!')
                torch.save(net.state_dict(), './leave_sub1.pth')
                return net

        else:
            print('trigger times: 0')
            trigger_times = 0

        last_loss = np.mean(val_losses)
        # -------------------end of early stopping
        # net.fc.train()  # reset to train mode after iterationg through validation data
        net.train()  # train the linear layer only

        net.encoder.eval()  # make the encoder not trainable
        for param in net.encoder.parameters():
            param.requires_grad = False

        print("Epoch: {}/{}...".format(e + 1, epochs),
              "Train Loss: {:.4f}...".format(np.mean(train_losses)),
              "Train Acc: {:.4f}...".format(train_acc / len(train_dl)),
              "Val Loss: {:.4f}...".format(np.mean(val_losses)),
              "Val Acc: {:.4f}...".format(accuracy),
              "F1-Score: {:.4f}...".format(f1score))
        res.append(f1score / len(val_dl))

        exp_lr_scheduler.step()  # lr decay

    print('model trained and saved!')
    torch.save(net.state_dict(), './adapted_model.pth')
    return net


CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()

train_loader = dl.run('train')
val_loader = dl.run('val')

model = AdaptModel()
model = model.cuda()

adapt_model = train(model, train_loader, val_loader, 500, args.batch_size, 0.03, delta=0.002, patience=10)


##########################################################################################
# final, test and plot confusion matrix
##########################################################################################

# evaluate and print predict result
def eval_perf(ground_truth, predicted_event, path):
    print('Accuracy score is: ')
    acc = accuracy_score(ground_truth, predicted_event)
    print(acc)
    print('Confusion Matrix is:')
    my_matrix = confusion_matrix(ground_truth, predicted_event)
    my_matrix_n = normalize(my_matrix, axis=1, norm='l1')
    print(pd.DataFrame(my_matrix_n).applymap(lambda x: '{:.2%}'.format(x)).values)

    target_names = ['walkF', 'walkL', 'walkR', 'upstairs', 'downstairs', 'run',
                    'jump', 'sit', 'stand', 'lying']
    df_cm = pd.DataFrame(my_matrix_n, index=[i for i in target_names],
                         columns=[i for i in target_names])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(path + str(args.num_samples) + '_samples_confusion_matrix.png')
    print(classification_report(ground_truth, predicted_event, target_names=target_names))
    return acc


def test(net, test_dl, batch_size, path):
    val_h = net.encoder.init_hidden(batch_size)
    outputs, y_true = [], []
    net.eval()

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_dl):
            inputs, targets = inputs.cuda(), labels.cuda()

            val_h = tuple([each.data for each in val_h])

            output = net(inputs.float(), val_h, batch_size)

            outputs.append(output.max(1).indices.detach().cpu().numpy())
            y_true.append(labels.detach().cpu().numpy())

        outputs = np.concatenate(outputs)
        y_true = np.concatenate(y_true)
        acc = eval_perf(y_true, outputs, path)
        return acc


test_loader = dl.run('test')
acc = test(adapt_model, test_loader, args.batch_size, PATH)
print(acc)
np.savetxt(PATH + str(args.num_samples) +'.txt', np.array(acc).reshape(-1,1))

