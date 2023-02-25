from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import argparse

NB_SENSOR_CHANNELS = 6
SLIDING_WINDOW_LENGTH = 200


############################################################################
# define the dataloader, load the train data (noisy in train)
#############################################################################
class uschad_dataset(Dataset):
    def __init__(self, r, noise_mode, mode, leave_sub):
        self.mode = mode
        self.r = r
        self.noise_mode = noise_mode

        if self.mode == 'train':
            X_train, y_train = [], []
            for subid in range(1, 15):
                if subid == leave_sub:
                    continue
                x_, y_ = [], []
                for tid in range(1, 5):
                    xtrain = self.load_tensor(subid, tid)
                    # ytrain = self.load_label(subid, tid)
                    ytrain = self.load_noisy_label_per_trial(subid, tid)
                    x_.append(xtrain)
                    y_.append(ytrain)
                # print(len(X_train))
                x_ = np.concatenate(x_)
                y_ = np.concatenate(y_)
                X_train.append(x_)
                y_train.append(y_)
                print(y_.shape)

        elif self.mode == 'val':
            X_train, y_train = [], []
            for subid in range(1, 15):
                X_train.append(self.load_tensor(subid, 5))
                y_train.append(self.load_label(subid, 5))

        self.data = X_train
        self.label = y_train
        self.make_same_length()

    def make_same_length(self):
        length = [y_.shape[0] for y_ in self.label]
        MAX_LENGTH = max(length)
        for i in range(13):
            dt = self.data[i]
            lb = self.label[i]
            num_sample = dt.shape[0]
            if MAX_LENGTH == num_sample: continue
            indices = np.random.permutation(num_sample)[:int(MAX_LENGTH - num_sample)]
            dt_ = np.vstack((dt[indices], dt))
            lb_ = np.concatenate((lb[indices], lb))
            assert dt_.shape[0] == lb_.shape[0]
            self.data[i] = dt_
            self.label[i] = lb_

    def load_tensor(self, name, tid):
        # Read the array from disk
        new_data = np.loadtxt(
            "E:/2022_spring/divideMix/data/processed_uschad/Sub" + str(name) + "_t" + str(tid) + "_data.txt")
        # new_data = np.loadtxt(
        #     "../data/processed_uschad/Sub" + str(name) + "_t" + str(tid) + "_data.txt")

        # Note that this returned a 2D array!
        # print (new_data.shape)

        # However, going back to 3D is easy if we know the
        # original shape of the array
        new_data = new_data.reshape((-1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
        return new_data

    def load_label(self, name, tid):
        y = np.loadtxt(
            'E:/2022_spring/divideMix/data/processed_uschad/Sub' + str(name) + '_t' + str(tid) + '_label.txt')
        # y = np.loadtxt(
        #     '../data/processed_uschad/Sub' + str(name) + '_t' + str(tid) + '_label.txt')
        y = y - 1
        return y

    def load_noisy_label(self, name):
        noise_file = './uschad/' + str(self.r) + '_' + self.noise_mode + '_' + str(name) + '.json'
        if os.path.exists(noise_file):
            noise_label = json.load(open(noise_file, "r"))
        return np.array(noise_label)

    def load_noisy_label_per_trial(self, name, tid):
        noise_file = './uschad/' + str(self.r) + '_' + self.noise_mode + '_s' + str(name) + 't' + str(tid) + '.json'
        if os.path.exists(noise_file):
            noise_label = json.load(open(noise_file, "r"))
        return np.array(noise_label)

    def __getitem__(self, index):
        x_list = [self.data[i][index] for i in range(13)]
        y_list = [self.label[i][index] for i in range(13)]
        # return self.data[index], self.label[index]
        return x_list, y_list, index

    def __len__(self):
        length = [y_.shape[0] for y_ in self.label]
        return min(length)


# class uschad_dataloader(): #dataloader for feature extractor training
#     def __init__(self, batch_size, leave_sub, num_workers):
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         train_ds,val_ds = [], []
#         for i in range(1, 15):
#             if i == leave_sub:
#                 continue
#             train_ds.append(uschad_dataset(mode='train',subid=i))
#             val_ds.append(uschad_dataset(mode='val',subid=i))
#         self.dts = train_ds
#         self.edts = val_ds
#         self.train_dataset = ConcatDataset(train_ds) #merge all source into one, and sample batches from
#         self.eval_dataset = ConcatDataset(val_ds)
#
#     def run(self, mode, subid=1):
#         if mode == 'all_train':
#             train_loader = DataLoader(dataset=self.train_dataset,
#                                       batch_size=self.batch_size,
#                                       shuffle=True,
#                                       num_workers=self.num_workers,
#                                       drop_last=True)
#             return train_loader
#
#         elif mode == 'all_val':
#             val_loader = DataLoader(dataset=self.eval_dataset,
#                                     batch_size=self.batch_size,
#                                     shuffle=True,
#                                     num_workers=self.num_workers,
#                                     drop_last=True)
#             return val_loader
#         elif mode == 'train_sub':
#             train_loader = DataLoader(dataset=self.dts[subid],
#                                       batch_size=self.batch_size,
#                                       shuffle=True,
#                                       num_workers=self.num_workers,
#                                       drop_last=True)
#             return train_loader
#         elif mode == 'val_sub':
#             val_loader = DataLoader(dataset=self.edts[subid],
#                                       batch_size=self.batch_size,
#                                       shuffle=True,
#                                       num_workers=self.num_workers,
#                                       drop_last=True)
#             return val_loader

class uschad_dataloader():  # dataloader for feature extractor training
    def __init__(self, r, noise_mode, batch_size, leave_sub, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = uschad_dataset(r, noise_mode, 'train',
                                            leave_sub)  # merge all source into one, and sample batches from
        self.eval_dataset = uschad_dataset(r, noise_mode, 'val', leave_sub)

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
