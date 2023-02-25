'''
write my basic bmtl model training into pytorch

define model, dataloader, loss all in one file
use pretrained self-supervised feature extractor as F
'''
import collections
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
import argparse
from bmtl_loss import MultiTaskLossWrapper, get_accuracy, get_accuracy_soft
from bmtl_model import MultiTaskModel
from dataloader_uschad import uschad_dataloader
from random import randint

parser = argparse.ArgumentParser(description='PyTorch USCHAD Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.03, type=float, help='initial learning rate')
parser.add_argument('--noise_mode', default='asym')
parser.add_argument('--alpha', default=0.2, type=float, help='parameter for Beta')
parser.add_argument('--r', default=0.4, type=float, help='noise ratio')
parser.add_argument('--leave_sub', default=14, type=int, help='leave sub id')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--dataset', default='uschad', type=str)
parser.add_argument('--num_workers', default=0, type=int)
args = parser.parse_args()

HIDDEN_UNIT = 128
NUM_CLASS = 10
NB_SENSOR_CHANNELS = 6
SLIDING_WINDOW_LENGTH = 200
BATCH_SIZE = 64
device = 'cuda'
n_epochs = 200
LR = 0.0001
NUM_WORKERS = 0
NUM_SUBS = 13

model = MultiTaskModel()
model = model.cuda()
print(model)



dl = uschad_dataloader(r=args.r, noise_mode=args.noise_mode,
                       batch_size=BATCH_SIZE, leave_sub=args.leave_sub, num_workers=NUM_WORKERS)
# initialize dataloader for feature extractor here, leave the dataloaders for branches in the train func
# test b = next(iter(train_dl))
train_dl = dl.run('train')
val_dl = dl.run('val')


############################################################################
# define the training process
#############################################################################
def make_inputs(inputs, labels):
    res, targets = [], []
    for i in range(13):
        targets.append(labels[i].long().to(device))
        res.append(inputs[i].to(device))
    return res, targets


def train_branches(epoch, net, optimizer, train_dl, loss_func):
    net.train()
    net.encoder.eval()
    # initialize hidden state
    h = net.encoder.init_hidden(BATCH_SIZE)

    # freeze feature extractor, unfreeze others
    ct = 0  # for my model, ct will count to 14
    for child in net.children():
        ct += 1
        if ct < 2:
            for param in child.parameters():
                param.requires_grad = False
        else:
            for param in child.parameters():
                param.requires_grad = True

    # train branches
    num_iter = (len(train_dl.dataset.data[0]) // BATCH_SIZE) + 1
    for batch_idx, (inputs, labels, indices) in enumerate(train_dl):
        inputs, targets = make_inputs(inputs, labels)
        h = tuple([each.data for each in h])
        optimizer.zero_grad()
        outputs = net(inputs, h)
        # loss = loss_func(net, outputs, targets[0], targets[1], targets[2], targets[3], targets[4], targets[5],
        #                  targets[6],
        #                  targets[7], targets[8], targets[9], targets[10], targets[11], targets[12])
        loss = loss_func(net, outputs, targets, indices, soft=False, elr=True)

        loss.backward()
        optimizer.step()

        print('\r', '%s: | Epoch [%3d/%3d] Train S Iter[%3d/%3d]\t BMTL-loss: %.4f '
              % ('uschad', epoch, n_epochs, batch_idx + 1, num_iter,
                 loss.item()), end=''
              )

    accs = []
    for i in range(13):
        accs.append(get_accuracy(outputs[i], targets[i]))

    print(  # print the acc result only once per epoch
        'accuracy_1: %.4f\t accuracy_2: %.4f\t accuracy_3: %.4f\t accuracy_4: %.4f\n accuracy_5: %.4f\t accuracy_6: %.4f\t accuracy_7: %.4f\t accuracy_8: %.4f\n accuracy_9: %.4f\t accuracy_10: %.4f\t accuracy_11: %.4f\t accuracy_12: %.4f\t accuracy_13: %.4f\n'
        % (accs[0], accs[1], accs[2], accs[3], accs[4], accs[5], accs[6], accs[7], accs[8], accs[9], accs[10], accs[11],
           accs[12]
           ))

    train_log.write('%s: | Epoch [%3d/%3d] Train S Iter[%3d/%3d]\t BMTL-loss: %.4f \n'
                    % ('uschad', epoch, n_epochs, batch_idx + 1, num_iter,
                       loss.item()))
    train_log.write(
        'accuracy_1: %.4f\t accuracy_2: %.4f\t accuracy_3: %.4f\t accuracy_4: %.4f\n accuracy_5: %.4f\t accuracy_6: %.4f\t accuracy_7: %.4f\t accuracy_8: %.4f\n accuracy_9: %.4f\t accuracy_10: %.4f\t accuracy_11: %.4f\t accuracy_12: %.4f\t accuracy_13: %.4f\n'
        % (accs[0], accs[1], accs[2], accs[3], accs[4], accs[5], accs[6], accs[7], accs[8], accs[9], accs[10], accs[11],
           accs[12]
           ))
    train_log.flush()

def mixUp1(inputs, labels, batch_size):
    #v1: mix across both subs and classes
    l = np.random.beta(args.alpha, args.alpha)
    l = max(l, 1 - l)
    # random sub pairs to be mixed up
    id_pairs, mix_inputs, mix_labels = [], [], []
    for i in range(13):
        # generate random subid pairs
        id_pairs.append((i, randint(0, 12)))

        # sample and mix all sub pairs (i, id_pairs[i][1])
        idx = torch.randperm(inputs[i].size(0))
        j = id_pairs[i][1]
        input_a, input_b = inputs[i], inputs[j]  # inputs are mixed up with permuted inputs
        target_a, target_b = labels[i], labels[j]

        # idx_ = torch.where(target_a == target_b)[0].to(device)
        # if idx_.size(0) != inputs[i].size(0):
        #     # print('need to duplicate samples for idx_')
        #     idx_diff = torch.randint(0, idx_.size(0), (inputs[i].size(0) - idx_.size(0),), device='cuda')
        #     idx_all = torch.cat((idx_, idx_diff))
        #     # print('new idx are: ', idx_all.size)

        #keep source unchange, random permute target and mixup
        input_a, input_b = input_a, input_b[idx]
        target_a, target_b = target_a, target_b[idx]
        target_a = torch.zeros(batch_size, args.num_class).scatter_(1, target_a.long().view(-1, 1), 1)
        target_b = torch.zeros(batch_size, args.num_class).scatter_(1, target_b.long().view(-1, 1), 1)

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        mix_inputs.append(mixed_input.to(device))
        mix_labels.append(mixed_target.to(device))

    return mix_inputs, mix_labels


def mixUp2(inputs, labels, batch_size):
    # v2: mix across sub but within the same class
    l = np.random.beta(args.alpha, args.alpha)
    l = max(l, 1 - l)
    # random sub pairs to be mixed up
    id_pairs, mix_inputs, mix_labels = [], [], []
    for i in range(13):
        # generate random subid pairs
        id_pairs.append((i, randint(0, 12)))

        # sample and mix all sub pairs (i, id_pairs[i][1])
        j = id_pairs[i][1]
        input_a, input_b = inputs[i], inputs[j]  # inputs are mixed up with permuted inputs
        target_a, target_b = labels[i], labels[j]

        target_b, idx = torch.sort(target_b)
        input_b = input_b[idx]
        k,v = target_b.unique(return_counts=True)
        nums = {}
        for key in range(10): nums[key] = -1
        temp, iter1, iter2 = 0, 0, 0
        while iter2 < 10 and iter1 < len(k):
            if k[iter1].item() != iter2:
                iter2 += 1

            else:
                temp += v[iter1].item()
                nums[iter2] = temp
                iter2 += 1
                iter1 += 1

        idx_ = []
        for t in target_a:
            t_ = t.item()
            i2 = nums[t_]
            if i2 == -1:
                idx_.append(randint(0, batch_size - 1))
                continue
            elif t_ == 0:
                idx_.append(randint(0,nums[t_]))
            else:
                i1 = nums[int(t_-1)]
                idx_.append(randint(i1, i2-1))

        # keep source unchange, random permute target and mixup
        input_a, input_b = input_a, input_b[idx_]
        target_a, target_b = target_a, target_b[idx_]
        target_a = torch.zeros(batch_size, args.num_class).scatter_(1, target_a.long().view(-1, 1), 1)
        target_b = torch.zeros(batch_size, args.num_class).scatter_(1, target_b.long().view(-1, 1), 1)

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b #no effect here, mixed_target actually still = target_a

        mix_inputs.append(mixed_input.to(device))
        mix_labels.append(mixed_target.to(device))

    return mix_inputs, mix_labels


def train_feature_extractor(epoch, net, optimizer, train_dl, loss_func, batch_size):
    net.eval()
    net.encoder.train()
    # initialize hidden state
    h = net.encoder.init_hidden(batch_size)

    # freeze branches, unfreeze feature extractor
    ct = 0
    for child in net.children():
        ct += 1
        if ct < 2:
            for param in child.parameters():
                param.requires_grad = True
        else:
            for param in child.parameters():
                param.requires_grad = False

    # train branches
    num_iter = (len(train_dl.dataset) // batch_size) + 1
    for batch_idx, (inputs, labels, indices) in enumerate(train_dl): #indices to calculate ELR term
        # inputs, targets = make_inputs(inputs, labels)
        inputs, targets = mixUp1(inputs, labels, batch_size)

        h = tuple([each.data for each in h])

        optimizer.zero_grad()
        outputs = net(inputs, h)

        loss = loss_func(net, outputs, targets, indices, soft=True, elr=True)

        loss.backward()
        optimizer.step()
        # use '\r' to make print in just one line
        print('\r', '%s: | Epoch [%3d/%3d] Train F Iter[%3d/%3d]\t BMTL-loss: %.4f '
              % ('uschad', epoch, n_epochs, batch_idx + 1, num_iter,
                 loss.item()), end=''
              )

    accs = []
    for i in range(13):
        accs.append(get_accuracy_soft(outputs[i], targets[i]))

    print(
        'accuracy_1: %.4f\t accuracy_2: %.4f\t accuracy_3: %.4f\t accuracy_4: %.4f\n accuracy_5: %.4f\t accuracy_6: %.4f\t accuracy_7: %.4f\t accuracy_8: %.4f\n accuracy_9: %.4f\t accuracy_10: %.4f\t accuracy_11: %.4f\t accuracy_12: %.4f\t accuracy_13: %.4f\n'
        % (accs[0], accs[1], accs[2], accs[3], accs[4], accs[5], accs[6], accs[7], accs[8], accs[9], accs[10], accs[11],
           accs[12]
           ))

    train_log.write('%s: | Epoch [%3d/%3d] Train F Iter[%3d/%3d]\t BMTL-loss: %.4f \n'
                    % ('uschad', epoch, n_epochs, batch_idx + 1, num_iter,
                       loss.item()))
    train_log.write(
        'accuracy_1: %.4f\t accuracy_2: %.4f\t accuracy_3: %.4f\t accuracy_4: %.4f\n accuracy_5: %.4f\t accuracy_6: %.4f\t accuracy_7: %.4f\t accuracy_8: %.4f\n accuracy_9: %.4f\t accuracy_10: %.4f\t accuracy_11: %.4f\t accuracy_12: %.4f\t accuracy_13: %.4f\n'
        % (accs[0], accs[1], accs[2], accs[3], accs[4], accs[5], accs[6], accs[7], accs[8], accs[9], accs[10], accs[11],
           accs[12]
           ))
    train_log.flush()


def val(epoch, net, val_dl, loss_func):
    net.eval()
    # initialize hidden state
    h = net.encoder.init_hidden(BATCH_SIZE)
    with torch.no_grad():
        losses = []
        for batch_idx, (inputs, labels, _) in enumerate(val_dl):
            inputs, targets = make_inputs(inputs, labels)

            outputs = net(inputs, h)

            # loss = loss_func(net, outputs, targets[0], targets[1], targets[2], targets[3], targets[4], targets[5],
            #                  targets[6],
            #                  targets[7], targets[8], targets[9], targets[10], targets[11], targets[12])
            loss = loss_func(net, outputs, targets)
            losses.append(loss.item())

        print("\n| Test Epoch #%d\t loss: %.4f\n" % (epoch, np.mean(losses)))
        accs = []
        for i in range(13):
            accs.append(get_accuracy(outputs[i], targets[i]))

        print(
            'accuracy_1: %.4f\t accuracy_2: %.4f\t accuracy_3: %.4f\t accuracy_4: %.4f\n accuracy_5: %.4f\t accuracy_6: %.4f\t accuracy_7: %.4f\t accuracy_8: %.4f\n accuracy_9: %.4f\t accuracy_10: %.4f\t accuracy_11: %.4f\t accuracy_12: %.4f\t accuracy_13: %.4f\n'
            % (accs[0], accs[1], accs[2], accs[3], accs[4], accs[5], accs[6], accs[7], accs[8], accs[9], accs[10],
               accs[11], accs[12]
               ))
        test_log.write("\n| Test Epoch #%d\t loss: %.4f\n" % (epoch, np.mean(losses)))
        test_log.write(
            'accuracy_1: %.4f\t accuracy_2: %.4f\t accuracy_3: %.4f\t accuracy_4: %.4f\n accuracy_5: %.4f\t accuracy_6: %.4f\t accuracy_7: %.4f\t accuracy_8: %.4f\n accuracy_9: %.4f\t accuracy_10: %.4f\t accuracy_11: %.4f\t accuracy_12: %.4f\t accuracy_13: %.4f\n'
            % (accs[0], accs[1], accs[2], accs[3], accs[4], accs[5], accs[6], accs[7], accs[8], accs[9], accs[10],
               accs[11], accs[12]
               ))
        test_log.flush()


# optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
optimizer1 = optim.Adam(model.encoder.parameters(), lr=0.0001, betas=(0.9, 0.999))
optimizer2 = optim.Adam([
    {'params': model.fc1.parameters()}, {'params': model.fc2.parameters()}, {'params': model.fc3.parameters()},
    {'params': model.fc4.parameters()},
    {'params': model.fc5.parameters()}, {'params': model.fc6.parameters()}, {'params': model.fc7.parameters()},
    {'params': model.fc8.parameters()},
    {'params': model.fc9.parameters()}, {'params': model.fc10.parameters()}, {'params': model.fc11.parameters()},
    {'params': model.fc12.parameters()}, {'params': model.fc13.parameters()}
], lr=0.0001, betas=(0.9, 0.999))

# make a log files to save the train and val acc and loss per each epoch
now = datetime.now()
date_time_str = now.strftime("%Y-%m-%d")
test_log = open('../checkpoint/val_' + date_time_str + str(args.leave_sub) +'_'+str(args.r)+'_'+args.noise_mode + '_log.txt', 'w')
train_log = open('../checkpoint/train_' + date_time_str + str(args.leave_sub) +'_'+str(args.r)+'_'+args.noise_mode + '_log.txt', 'w')

#initialize global variable for ELR loss (need NUM_SUBS)
NUM_EXAMP = len(train_dl.dataset)
loss_func = MultiTaskLossWrapper(NUM_EXAMP, NUM_CLASS, NUM_SUBS).to(device)  # just making sure the loss is on the gpu

for epoch in range(n_epochs):
    lr = LR
    if epoch >= 100:
        lr /= 10
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    train_branches(epoch, model, optimizer2, train_dl, loss_func)
    train_feature_extractor(epoch, model, optimizer1, train_dl, loss_func, BATCH_SIZE)
    val(epoch, model, val_dl, loss_func)

    if epoch % 10 == 0:
        torch.save(model.state_dict(), 'bmtl_model_leave' + str(args.leave_sub) +'_'+str(args.r)+'_'+args.noise_mode)

# save the trained models
torch.save(model.state_dict(), 'bmtl_model_leave' + str(args.leave_sub) +'_'+str(args.r)+'_'+args.noise_mode)
