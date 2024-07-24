import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import collections
from functools import reduce
from torch.autograd import Variable
from torch import nn
import copy
from model.accuracy import *
from config import max_iter

LOSS_ACC_BATCH_SIZE = 128   # When computing loss and accuracy, use blocks of LOSS_ACC_BATCH_SIZE


class Model():
    def __init__(self, rand_seed=None, learning_rate=0.001, model_name=None, device=None,
                 flatten_weight=False, pretrained_model_file = None):
        super(Model, self).__init__()
        if device is None:
            raise Exception('Device not specified in Model()')
        if rand_seed is not None:
            torch.manual_seed(rand_seed)
        self.model = None
        self.loss_fn = None
        self.weights_key_list = None
        self.weights_size_list = None
        self.weights_num_list = None
        self.optimizer = None
        self.flatten_weight = flatten_weight
        self.learning_rate = learning_rate
#Need to create all the models shown below
        if model_name == 'ModelFMnist':
            from model.logistic import LogisticRegression_F
            self.model = LogisticRegression_F().to(device)
        elif model_name == 'ModelCifar10':
            from model.cnn import ModelCNNcifar
            self.model = ModelCNNcifar().to(device)
        elif model_name == 'ModelEMnist':
            from model.two_nn import Two_NN
            self.model = Two_NN().to(device)
        elif model_name == 'ModelShakespeare':
            from model.rnn import RNN
            self.model = RNN().to(device)
        elif model_name == 'Synthetic': #The input params are different for Synthetic dataset so just construct different classes
            from model.logistic import LogisticRegression_S
            self.model = LogisticRegression_S().to(device)

        if pretrained_model_file is not None:
            self.model.load_state_dict(torch.load(pretrained_model_file, map_location=device))

        
#         if model_name == 'LogisticRegression':
#             weight_decay = 0
#             self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0, weight_decay=weight_decay)
#         else:
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
#         if model_name == 'ModelCifar10':
#             self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[int(0.5 * max_iter), int(0.75 * max_iter)], gamma=0.1)
#         else:
#             self.scheduler = None
#         self.model.to(device)
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        if model_name == 'ModelShakespeare':
            torch.nn.CrossEntropyLoss(ignore_index=0)
        else:
            self.loss_fn = nn.CrossEntropyLoss().to(device)
        self._get_weight_info()

    def _get_weight_info(self):
        self.weights_key_list = []
        self.weights_size_list = []
        self.weights_num_list = []
        state = self.model.state_dict()
        for k, v in state.items():
            shape = list(v.size())
            self.weights_key_list.append(k)
            self.weights_size_list.append(shape)
            if len(shape) > 0:
                num_w = reduce(lambda x, y: x * y, shape)
            else:
                num_w=0
            self.weights_num_list.append(num_w)

    def get_weight_dimension(self):
        dim = sum(self.weights_num_list)
        return dim

    def get_weight(self):
        with torch.no_grad():
            state = self.model.state_dict()
            if self.flatten_weight:
                weight_flatten_tensor = torch.Tensor(sum(self.weights_num_list)).to(state[self.weights_key_list[0]].device)
                start_index = 0
                for i,[_, v] in zip(range(len(self.weights_num_list)), state.items()):
                    weight_flatten_tensor[start_index:start_index+self.weights_num_list[i]] = v.view(1, -1)
                    start_index += self.weights_num_list[i]

                return weight_flatten_tensor
            else:
                return copy.deepcopy(state)

    def assign_weight(self, w):
        if self.flatten_weight:
            self.assign_flattened_weight(w)
        else:
            self.model.load_state_dict(w)

    def assign_flattened_weight(self, w):

        weight_dic = collections.OrderedDict()
        start_index = 0

        for i in range(len(self.weights_key_list)):
            sub_weight = w[start_index:start_index+self.weights_num_list[i]]
            if len(sub_weight) > 0:
                weight_dic[self.weights_key_list[i]] = sub_weight.view(self.weights_size_list[i])
            else:
                weight_dic[self.weights_key_list[i]] = torch.tensor(0)
            start_index += self.weights_num_list[i]
        self.model.load_state_dict(weight_dic)

    def _data_reshape(self, imgs, labels=None):
        if len(imgs.size()) < 3:
            x_image = imgs.view([-1, self.channels, self.img_size, self.img_size])
            if labels is not None:
                _, y_label = torch.max(labels.data, 1)  # From one-hot to number
            else:
                y_label = None
            return x_image, y_label
        else:
            return imgs, labels

    def accuracy(self, data_test_loader, w, device, is_shakespeare=False):
        if w is not None:
            self.assign_weight(w)

        self.model.eval()
        total_count = 0
        total_correct = 0
        avg_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_test_loader):
                images, labels = Variable(images).to(device), Variable(labels).to(device)
                output = self.model(images)
                avg_loss += self.loss_fn(output, labels).sum()
                pred = output.data.max(1)[1]
#                 total_correct += pred.eq(labels.data.view_as(pred)).sum()
#         avg_loss /= len(data_test_loader.dataset)
#         acc = float(total_correct) / len(data_test_loader.dataset)
                if is_shakespeare:
                    correct, count = stream_accuracy(output, labels)
                else:
                    correct, count = classification_accuracy(output, labels)
                total_correct += correct
                total_count += count
        avg_loss /= len(data_test_loader.dataset)
        acc = float(total_correct) / total_count
        
        return avg_loss.item(), acc
    
    def accuracy_im(self, images_list, labels_list, w, device):
        if w is not None:
            self.assign_weight(w)

        self.model.eval()

        total_correct = 0
        avg_loss = 0.0
        with torch.no_grad():
            for i, (image, label) in enumerate(zip(images_list, labels_list)):
                image, label = Variable(image.unsqueeze(0)).to(device), Variable(label.unsqueeze(0)).to(device)
                output = self.model(image)
                avg_loss += self.loss_fn(output, label).sum()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(label.data.view_as(pred)).sum()
        avg_loss /= len(images_list)
        acc = float(total_correct) / len(images_list)

        return avg_loss.item(), acc