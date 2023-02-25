import torch.nn as nn
from deepconvlstm import HARModel
import torch

HIDDEN_UNIT = 128
NUM_CLASS = 10
NB_SENSOR_CHANNELS = 6
SLIDING_WINDOW_LENGTH = 200
BATCH_SIZE = 64
device = 'cuda'
NUM_WORKERS = 0


############################################################################
# define the model
#############################################################################

class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        self.encoder = self.load_feature_extractor()
        # self.encoder = HARModel()
        self.fc1 = nn.Linear(HIDDEN_UNIT,
                             NUM_CLASS)  # ps is the drop-out ratio, can be single value for the first layer or a list for each
        self.fc2 = nn.Linear(HIDDEN_UNIT, NUM_CLASS)
        self.fc3 = nn.Linear(HIDDEN_UNIT, NUM_CLASS)
        self.fc4 = nn.Linear(HIDDEN_UNIT, NUM_CLASS)
        self.fc5 = nn.Linear(HIDDEN_UNIT, NUM_CLASS)
        self.fc6 = nn.Linear(HIDDEN_UNIT, NUM_CLASS)
        self.fc7 = nn.Linear(HIDDEN_UNIT, NUM_CLASS)
        self.fc8 = nn.Linear(HIDDEN_UNIT, NUM_CLASS)
        self.fc9 = nn.Linear(HIDDEN_UNIT, NUM_CLASS)
        self.fc10 = nn.Linear(HIDDEN_UNIT, NUM_CLASS)
        self.fc11 = nn.Linear(HIDDEN_UNIT, NUM_CLASS)
        self.fc12 = nn.Linear(HIDDEN_UNIT, NUM_CLASS)
        self.fc13 = nn.Linear(HIDDEN_UNIT, NUM_CLASS)

    def load_feature_extractor(self):
        model_path = 'self_supervision_feature_extractor'
        model = HARModel()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def forward(self, x, hidden, batch_size=BATCH_SIZE):
        '''
        need the num_task here to do forward to different branches
        '''
        # depend on type of transform, output binary class
        x0, _ = self.encoder(x[0].float(), hidden, batch_size)
        x0 = self.fc1(x0)
        x0 = x0.view(batch_size, -1, NUM_CLASS)[:, -1, :]
        t1 = x0

        x1, _ = self.encoder(x[1].float(), hidden, batch_size)
        x1 = self.fc2(x1)
        x1 = x1.view(batch_size, -1, NUM_CLASS)[:, -1, :]
        t2 = x1

        x2, _ = self.encoder(x[2].float(), hidden, batch_size)
        x2 = self.fc3(x2)
        x2 = x2.view(batch_size, -1, NUM_CLASS)[:, -1, :]
        # t3 = torch.sigmoid(x2) #no sigmoid here!
        t3 = x2

        x3, _ = self.encoder(x[3].float(), hidden, batch_size)
        x3 = self.fc4(x3)
        x3 = x3.view(batch_size, -1, NUM_CLASS)[:, -1, :]
        t4 = x3

        x4, _ = self.encoder(x[4].float(), hidden, batch_size)
        x4 = self.fc5(x4)
        x4 = x4.view(batch_size, -1, NUM_CLASS)[:, -1, :]
        t5 = x4

        x5, _ = self.encoder(x[5].float(), hidden, batch_size)
        x5 = self.fc6(x5)
        x5 = x5.view(batch_size, -1, NUM_CLASS)[:, -1, :]
        t6 = x5

        x6, _ = self.encoder(x[6].float(), hidden, batch_size)
        x6 = self.fc7(x6)
        x6 = x6.view(batch_size, -1, NUM_CLASS)[:, -1, :]
        t7 = x6

        x7, _ = self.encoder(x[7].float(), hidden, batch_size)
        x7 = self.fc8(x7)
        x7 = x7.view(batch_size, -1, NUM_CLASS)[:, -1, :]
        t8 = x7

        x8, _ = self.encoder(x[8].float(), hidden, batch_size)
        x8 = self.fc9(x8)
        x8 = x8.view(batch_size, -1, NUM_CLASS)[:, -1, :]
        t9 = x8

        x9, _ = self.encoder(x[9].float(), hidden, batch_size)
        x9 = self.fc10(x9)
        x9 = x9.view(batch_size, -1, NUM_CLASS)[:, -1, :]
        t10 = x9

        x10, _ = self.encoder(x[10].float(), hidden, batch_size)
        x10 = self.fc11(x10)
        x10 = x10.view(batch_size, -1, NUM_CLASS)[:, -1, :]
        t11 = x10

        x11, _ = self.encoder(x[11].float(), hidden, batch_size)
        x11 = self.fc12(x11)
        x11 = x11.view(batch_size, -1, NUM_CLASS)[:, -1, :]
        t12 = x11

        x12, _ = self.encoder(x[12].float(), hidden, batch_size)
        x12 = self.fc13(x12)
        x12 = x12.view(batch_size, -1, NUM_CLASS)[:, -1, :]
        t13 = x12

        return [t1, t2, t3, t4, \
                t5, t6, t7, t8, \
                t9, t10, t11, t12, t13]


