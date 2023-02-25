import numpy as np
import argparse
import json
import random

WINDOW_LENGTH = 200
NUM_CHANNEL = 6

transition = {0: 5, 1: 2, 2: 1, 3: 1, 4: 2, 5: 3, 6: 5, 7: 9, 8: 6,
              9: 7}  # class transition for asymmetric noise

parser = argparse.ArgumentParser(description='PyTorch USCHAD Training')
parser.add_argument('--noise_mode', default='sym')
parser.add_argument('--r', default=0.6, type=float, help='noise ratio')
parser.add_argument('--dataset', default='uschad', type=str)
args = parser.parse_args()


def load_tensor(name, tid):
    # Read the array from disk
    new_data = np.loadtxt("../data/processed_uschad/Sub" + str(name) + "_t" + str(tid) + "_data.txt")

    # Note that this returned a 2D array!
    # print (new_data.shape)

    # However, going back to 3D is easy if we know the
    # original shape of the array
    new_data = new_data.reshape((-1, WINDOW_LENGTH, NUM_CHANNEL))
    return new_data


def load_label(name, tid):
    y = np.loadtxt('../data/processed_uschad/Sub' + str(name) + '_t' + str(tid) + '_label.txt')
    y = y - 1
    return y


noise_file = '%s/%.1f_%s' % (args.dataset, args.r, args.noise_mode)

for subid in range(1, 15):
    for tid in range(1, 6):
        path = noise_file + '_s' + str(subid) + 't' + str(tid) + '.json'

        train_data = load_tensor(subid, tid)
        train_label = load_label(subid, tid)

        TRAIN_SIZE = train_data.shape[0]
        noise_label = []
        idx = list(range(TRAIN_SIZE))
        random.shuffle(idx)
        num_noise = int(args.r * TRAIN_SIZE)
        noise_idx = idx[:num_noise]
        for i in range(TRAIN_SIZE):
            if i in noise_idx:
                if args.noise_mode == 'sym':
                    noiselabel = random.randint(0, 9)
                    noise_label.append(noiselabel)
                elif args.noise_mode == 'asym':
                    noiselabel = transition[train_label[i]]
                    noise_label.append(noiselabel)
            else:
                noise_label.append(int(train_label[i].tolist()))
        json.dump(noise_label, open(path, "w"))
