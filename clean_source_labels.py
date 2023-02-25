'''
train a model, make prediction with noisy sources, check the acc
how good the model can do to clean the source labels

'''


import numpy as np
import json
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
import argparse
from sklearn.manifold import TSNE

import matplotlib as mpl

mpl.use('Qt5Agg')

parser = argparse.ArgumentParser(description='PyTorch USCHAD Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.03, type=float, help='initial learning rate')
parser.add_argument('--noise_mode', default='sym')
parser.add_argument('--r', default=0.4, type=float, help='noise ratio')
parser.add_argument('--leave_sub', default=14, type=int, help='leave sub id')
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--num_samples', default=20, type=int)
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
# load trained model
# =============================================

def load_pretrain():
    model_path = PATH + '/bmtl_model_leave' + str(args.leave_sub) + '_' + str(args.r) + '_' + args.noise_mode
    model = MultiTaskModel()
    model.load_state_dict(torch.load(model_path))
    return model

model = load_pretrain()
model = model.cuda()
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
    # listact = ['walkF', 'walkL', 'walkR', 'upstairs', 'downstairs', 'run',
    #            'jump', 'sit', 'stand', 'lying']
    listact = ['correct','wrong']
    ind = 0
    for name, points in groups:
        f = listact[int(name - 1)]
        print(f)
        ax.scatter(points.x, points.y, label=f)
        ind += 1
    ax.legend()
    plt.show()
    # fig.savefig(PATH + str(args.num_samples) + '_samples_tsne.png')


##################################################################################
# still have to define a dataset and train, val, test dataloader first
################################################################################

class uschad_dataset(Dataset):
    def __init__(self, leave_sub):
        # load data for test
        X_test, y_test, y_noise, sub_ids = [], [],[],[]
        i=0
        for subid in range(1, 15):
            if subid == leave_sub:
                continue
            temp = []
            for tid in range(1, 4):
                X_test.append(self.load_tensor(subid, tid))  # id not in 1600,1635
                temp.append(self.load_label(subid, tid))
                y_noise.append(self.load_noisy_label(subid,tid,args.r,args.noise_mode))
            temp = np.concatenate(temp)
            sub_ids.append(np.ones_like(temp)*(i)) #change the id of coresponding output layer id
            i += 1
            y_test.append(temp)
        self.X_test = np.concatenate(X_test)
        self.y_test = np.concatenate(y_test)
        self.y_noise = np.concatenate(y_noise)
        self.sub_ids = np.concatenate(sub_ids)

    def load_tensor(self, name, tid):
        # Read the array from disk
        new_data = np.loadtxt(
            "E:/2022_spring/divideMix/data/processed_uschad/Sub" + str(name) + "_t" + str(tid) + "_data.txt")

        # Note that this returned a 2D array!
        # print (new_data.shape)

        # However, going back to 3D is easy if we know the
        # original shape of the array
        new_data = new_data.reshape((-1, SLIDING_WINDOW_LENGTH, NUM_CHANNEL))
        return new_data

    def load_label(self, name, tid):
        y = np.loadtxt(
            'E:/2022_spring/divideMix/data/processed_uschad/Sub' + str(name) + '_t' + str(tid) + '_label.txt')
        y = y - 1
        return y

    def load_noisy_label(self, name, tid, noise_rate, noise_mode):
        noise_file = './uschad/' + str(noise_rate) + '_' + noise_mode + '_s' + str(name) + 't' + str(tid) + '.json'
        noise_label = json.load(open(noise_file, "r"))
        return noise_label

    def __getitem__(self, index):
        return self.X_test[index], self.y_test[index], self.y_noise[index], self.sub_ids[index]

    def __len__(self):
        return len(self.y_test)


class uschad_dataloader():  # dataloader for feature extractor training
    def __init__(self, batch_size, leave_sub, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_dataset = uschad_dataset(leave_sub)

    def run(self, mode):
        if mode == 'test':
            test_loader = DataLoader(dataset=self.test_dataset,
                                     batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     drop_last=True)
            return test_loader


def make_inputs(inputs, labels):
    res, targets = [], []
    for i in range(13):
        # make target onehot vec
        targets.append(labels.long().to(device))
        res.append(inputs.to(device))
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

    outputs, ys, noise_lb = [], [], []
    with torch.no_grad():
        for batch_idx, (inputs, labels, noise_labels, _) in enumerate(all_loader):
            inputs, labels = inputs.cuda(), labels.cuda()

            output, h = m_encoder(inputs.float(), h, args.batch_size)
            output = output.view(args.batch_size, -1, 128)[:, -1, :]
            outputs.append(output.detach().cpu().numpy())
            ys.append(labels.detach().cpu().numpy())
            noise_lb.append(noise_labels)

    outputs = np.concatenate(outputs)
    ys = np.concatenate(ys)
    noise_lb = np.concatenate(noise_lb)
    y = np.array(ys==noise_lb)
    tsne(outputs, y)


all_loader = dl.run('test')
# plot_tsne(all_loader)

# initialize global variable for ELR loss (need NUM_SUBS)
NUM_EXAMP = len(all_loader.dataset)
loss_func = MultiTaskLossWrapper(NUM_EXAMP, args.num_class, NUM_SUBS).to(
    device)  # just making sure the loss is on the gpu


def get_accuracy(y_prob, y_true):
    _, preds = torch.max(y_prob, 1)
    assert y_true.size() == preds.size()
    return preds.eq(y_true).cpu().sum().item() / y_true.size(0)


CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()

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
    # plt.savefig(path + str(args.num_samples) + '_samples_confusion_matrix.png')
    print(classification_report(ground_truth, predicted_event, target_names=target_names))
    return acc


def test(net, test_dl, batch_size, path):
    val_h = net.encoder.init_hidden(batch_size)
    outputs, y_true, y_noise = [], [], []
    net.eval()

    with torch.no_grad():
        for batch_idx, (inputs, labels, noise_labels, subids) in enumerate(test_dl):
            inputs, targets = inputs.cuda(), labels.cuda()

            input_, target_ = make_inputs(inputs,targets)

            output = net(input_, val_h, batch_size)[subids[0].int()]

            outputs.append(output.max(1).indices.detach().cpu().numpy())
            y_true.append(labels.detach().cpu().numpy())
            y_noise.append(noise_labels)

        outputs = np.concatenate(outputs)
        y_true = np.concatenate(y_true)
        y_noise = np.concatenate(y_noise)
        acc = eval_perf(y_true, outputs, path)

        #test the noisy part, how much has been cleaned
        base = np.where(y_noise!=y_true)
        count = np.where((y_noise!=y_true) & (outputs==y_true))
        # print(base,count)
        return acc, len(count[0])/len(base[0])


test_loader = dl.run('test')
acc,ct = test(model, test_loader, args.batch_size, PATH)
print(acc,ct)

# write to file
with open("result.txt", "a") as myfile:
    myfile.write(str(args.r) + args.noise_mode + "\n")
    myfile.write(str(acc)+',')
    myfile.write(str(ct)+'\n')

