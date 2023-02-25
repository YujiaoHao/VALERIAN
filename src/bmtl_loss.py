import torch.nn as nn
import torch
import torch.nn.functional as F


############################################################################
# define loss and metrics
#############################################################################

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, num_examp, num_classes, task_num, beta=0.7, la=3):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))
        self.num_classes = num_classes
        self.target = self.make_target(num_examp)
        self.beta = beta
        self.lambda_ = la

    def make_target(self, num_examp):
        res = []
        for i in range(self.task_num):
            res.append(torch.zeros(num_examp, self.num_classes).cuda())
        return res

    def crossEntropy_with_prob(self, outputs_x, targets_x):
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        return Lx

    def elr(self, index, output, label, soft=False):
        final_loss = 0
        for i in range(self.task_num):
            y_pred = F.softmax(output[i], dim=1)
            y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
            y_pred_ = y_pred.data.detach()
            self.target[i][index] = self.beta * self.target[i][index] + (1 - self.beta) * (
                    (y_pred_) / (y_pred_).sum(dim=1, keepdim=True))
            if soft:
                ce_loss = self.crossEntropy_with_prob(output[i], label[i])
            else:
                ce_loss = F.cross_entropy(output[i], label[i])

            elr_reg = ((1 - (self.target[i][index] * y_pred).sum(dim=1)).log()).mean()
            final_loss += ce_loss + self.lambda_ * elr_reg
        return final_loss

    def forward(self, model, preds, lbs, index=[], soft=False, elr=False):
        '''
        model: the nn
        preds: list of predictions from multi-heads
        lbs: list of labels
        index: cal ELR
        soft: switch for calculate with long (False) or probability (True) form labels
        elr: calculate elr term or not (not useful when val or test model)
        '''
        if soft:
            crossEntropy = self.crossEntropy_with_prob
        else:
            crossEntropy = nn.CrossEntropyLoss()
        # loss0 = crossEntropy(preds[0], lb1.long())
        myloss = 0
        for i in range(self.task_num):
            myloss += crossEntropy(preds[i], lbs[i])

        miu = 0.4
        S0 = model.fc1.weight
        S1 = model.fc2.weight
        S2 = model.fc3.weight
        S3 = model.fc4.weight

        S4 = model.fc5.weight
        S5 = model.fc6.weight
        S6 = model.fc7.weight
        S7 = model.fc8.weight

        S8 = model.fc9.weight
        S9 = model.fc10.weight
        S10 = model.fc11.weight
        S11 = model.fc12.weight

        S12 = model.fc13.weight

        S = torch.hstack((S0, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12))
        #        myloss+= miu*np.linalg.norm(S[0].reshape(-1,5),ord=1)
        myloss += miu * torch.linalg.norm(torch.transpose(S, 0, 1), ord=1)

        # add elr loss to the loss term
        if elr:
            myloss += self.elr(index, preds, lbs, soft)
        return myloss


def get_accuracy(y_prob, y_true):
    _, preds = torch.max(y_prob, 1)
    assert y_true.size() == preds.size()
    return preds.eq(y_true).cpu().sum().item() / y_true.size(0)


def get_accuracy_soft(y_prob, y_true):
    _, preds = torch.max(y_prob, 1)
    _, gt = torch.max(y_true, 1)
    assert gt.size() == preds.size()
    return preds.eq(gt).cpu().sum().item() / gt.size(0)
