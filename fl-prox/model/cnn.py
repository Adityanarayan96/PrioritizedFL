import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn


# class ModelCNNcifar(nn.Module):
#     def __init__(self):
#         super(ModelCNNcifar, self).__init__()

#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels=3,
#                       out_channels=32,
#                       kernel_size=5,
#                       stride=1,
#                       padding=2,
#                       ),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(in_channels=32,
#                       out_channels=64,
#                       kernel_size=5,
#                       stride=1,
#                       padding=2,
#                       ),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )

#         self.fc1 = nn.Sequential(
#             nn.Linear(8 * 8 * 64, 512),
#             nn.ReLU(),
#             nn.Linear(512, 128),
#             nn.ReLU(),
#         )
#         self.fc2 = nn.Linear(128, 10)

#         # Use Kaiming initialization for layers with ReLU activation
#         @torch.no_grad()
#         def init_weights(m):
#             if type(m) == nn.Linear or type(m) == nn.Conv2d:
#                 torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#                 torch.nn.init.zeros_(m.bias)

#         self.conv.apply(init_weights)
#         self.fc1.apply(init_weights)

#     def forward(self, x):
#         conv_ = self.conv(x)
#         fc_ = conv_.view(-1, 8 * 8 * 64)
#         fc1_ = self.fc1(fc_)
#         output = self.fc2(fc1_)
#         return output


# class ModelCNNMnist(nn.Module):
#     def __init__(self):
#         super(ModelCNNMnist, self).__init__()

#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels=1,
#                       out_channels=32,
#                       kernel_size=5,
#                       stride=1,
#                       padding=2,
#                       ),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(in_channels=32,
#                       out_channels=32,
#                       kernel_size=5,
#                       stride=1,
#                       padding=2,
#                       ),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         self.fc1 = nn.Sequential(
#             nn.Linear(7 * 7 * 32, 128),
#             nn.ReLU(),
#         )
#         self.fc2 = nn.Linear(128, 10)

#         # Use Kaiming initialization for layers with ReLU activation
#         @torch.no_grad()
#         def init_weights(m):
#             if type(m) == nn.Linear or type(m) == nn.Conv2d:
#                 torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#                 torch.nn.init.zeros_(m.bias)

#         self.conv.apply(init_weights)
#         self.fc1.apply(init_weights)

#     def forward(self, x):
#         conv_ = self.conv(x)
#         fc_ = conv_.view(-1, 32*7*7)
#         fc1_ = self.fc1(fc_)
#         output = self.fc2(fc1_)
#         return output
# import torch
# import torch.nn as nn

class ModelCNNcifar(nn.Module):
    def __init__(self):
        super(ModelCNNcifar, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 8 * 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(128, 10)

        # Use Kaiming initialization for layers with ReLU activation
        @torch.no_grad()
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                torch.nn.init.zeros_(m.bias)

        self.conv.apply(init_weights)
        self.fc1.apply(init_weights)

    def forward(self, x):
        conv_ = self.conv(x)
        fc_ = conv_.view(-1, 8 * 8 * 64)
        fc1_ = self.fc1(fc_)
        output = self.fc2(fc1_)
        return output

# import torch
# import torch.nn as nn

# import torch
# import torch.nn as nn

# class ModelCNNcifar(nn.Module):
#     def __init__(self, in_channels=3, size=32, num_classes=10):
#         super(ModelCNNcifar, self).__init__()
#         self.in_channels = in_channels
#         self.height = self.width = size
#         self.num_classes = num_classes
#         self.conv2d_1 = torch.nn.Conv2d(self.in_channels, 32, kernel_size=5, padding=2)
#         self.max_pooling = nn.MaxPool2d(2, stride=2)
#         self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
#         self.flatten = nn.Flatten()
#         self.channels = 64*size//4*size//4
#         self.linear_1 = nn.Linear(self.channels, 512)
#         self.linear_2 = nn.Linear(512, 128)
#         self.linear_3 = nn.Linear(128, self.num_classes)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = x.reshape([-1, self.in_channels, self.height, self.width])
#         x = self.conv2d_1(x)
#         x = self.relu(x)
#         x = self.max_pooling(x)
#         x = self.conv2d_2(x)
#         x = self.relu(x)
#         x = self.max_pooling(x)
#         x = self.flatten(x)
#         x = self.linear_1(x)
#         x = self.relu(x)
#         x = self.linear_2(x)
#         x = self.relu(x)
#         x = self.linear_3(x)
#         return x
