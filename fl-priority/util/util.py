import numpy as np
import math
# from config import mixing_ratio,mixing_ratio_w,temp_worker_size
from config import iter_stop
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader, TensorDataset


# def create_data_loaders(train_data, test_data, batch_size_train=10, batch_size_test=10):
#     train_loader_list = []
#     worker_train_loader_list = []
#     data_iter_list = []
#     data_iter_list_w = []
#     all_train_data = []
#     all_train_labels = []

#     for user in train_data['users']:
#         user_data = train_data['user_data'][user]
#         x_train = torch.tensor(user_data['x'], dtype=torch.float32)
#         y_train = torch.tensor(user_data['y'], dtype=torch.long)
        
        
        
#         train_dataset = TensorDataset(x_train, y_train)
#         train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
#         train_loader_list.append(train_loader)
#         if 'f_' in user:  # user data
#             train_loader_list.append(train_loader)
#             data_iter_list.append(iter(train_loader))
#             all_train_data.append(x_train)
#             all_train_labels.append(y_train)
#         elif 'w_' in user:  # worker data
#             worker_train_loader_list.append(train_loader)
#             data_iter_list_w.append(iter(train_loader))
    
#     all_train_data = torch.cat(all_train_data, dim=0)
#     all_train_labels = torch.cat(all_train_labels, dim=0)
    
#     train_dataset = TensorDataset(all_train_data, all_train_labels)
#     data_train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    
#     all_test_data = []
#     all_test_labels = []

#     for user in test_data['users']:
#         user_data = test_data['user_data'][user]
#         x_test = torch.tensor(user_data['x'], dtype=torch.float32)
#         y_test = torch.tensor(user_data['y'], dtype=torch.long)

#         all_test_data.append(x_test)
#         all_test_labels.append(y_test)
    
# #     all_test_data = torch.tensor(test_data['x'], dtype=torch.float32)
# #     all_test_labels = torch.tensor(test_data['y'], dtype=torch.long)
#     all_test_data = torch.cat(all_test_data, dim=0)
#     all_test_labels = torch.cat(all_test_labels, dim=0)

#     test_dataset = TensorDataset(all_test_data, all_test_labels)
#     data_test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

#     return train_loader_list, data_iter_list, data_train_loader, data_test_loader, worker_train_loader_list, data_iter_list_w


# class NodeSampler:
#     def __init__(self, n_nodes, permutation=True):
#         self.n_nodes = n_nodes
#         self.permutation = permutation
#         self.remaining_permutation = []

#     def sample(self, node_sample_set, size):
#         if self.permutation:
#             sampled_set = []
#             while len(sampled_set) < size:
#                 if len(self.remaining_permutation) == 0:
#                     self.remaining_permutation = list(np.random.permutation(self.n_nodes))

#                 i = self.remaining_permutation.pop()

#                 if i in node_sample_set:
#                     sampled_set.append(i)
#         else:
#             sampled_set = np.random.choice(node_sample_set, size, replace=False)

#         return sampled_set


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
#         if isinstance(dataset, torch.utils.data.Subset):
#             self.targets = dataset.dataset.targets[idxs].numpy()
#         else:
#             self.targets = dataset.targets[idxs].numpy()

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def general_partition(data_train, n_nodes, n_shards, samples_per_shard, shards_per_client):
    dict_users = {i: np.array([], dtype='int64') for i in range(n_nodes)}
    if isinstance(data_train, torch.utils.data.Subset):
        indices = data_train.indices
        labels = data_train.dataset.targets[indices].numpy()
    else:
        labels = np.array(data_train.targets)
#     indices = data_train.indices
#     labels = data_train.labels
    unique_labels = np.unique(labels)
    
    shards = []
    
    # Create shards with single label
    for _ in range(n_shards):
        label = np.random.choice(unique_labels)
        label_indices = np.where(labels == label)[0]
        
        shard_indices = np.random.choice(label_indices, samples_per_shard, replace=True)
        shards.append(shard_indices)

    # Assign shards to clients
#     dict_users = {}
    for i in range(n_nodes):
        assigned_shard_indices = np.random.choice(len(shards), shards_per_client, replace=False)
        assigned_shards = [shards[idx] for idx in assigned_shard_indices]
        client_indices = np.concatenate(assigned_shards)
        
        np.random.shuffle(client_indices)
        
        dict_users[i] = client_indices.tolist()
    
    return dict_users

def fmnist_partition(data_train, n_nodes):
    return general_partition(data_train, n_nodes, 120, 500, 2)

def cifar_partition(data_train, n_nodes):
    return general_partition(data_train, n_nodes, 120, 500, 2)

def emnist_partition(data_train, n_nodes):
    return general_partition(data_train, n_nodes, 600, 180, 24)


def shake_partition(data_train, n_nodes):
    role_play_map = data_train.role_play_map
    texts = data_train.texts
    max_line_length = 80

    # Split lines into segments of 80 characters
    truncated_texts = {}
    for role, lines in texts.items():
        truncated_texts[role] = [line[i:i+max_line_length] for line in lines for i in range(0, len(line), max_line_length)]

    # Assign roles to clients
    dict_users = {}
    for i in range(n_nodes):
        role = random.choice(list(role_play_map.keys()))
        play = role_play_map[role]
        dict_users[i] = (role, play, truncated_texts[role])

    return dict_users

def allocate_priority(dict_users, pr_nodes):
    node_ids = list(dict_users.keys())
    pr_node_ids = np.random.choice(node_ids, pr_nodes, replace=False)
    pr_dict_users = {i: dict_users[i] for i in pr_node_ids}
    fr_dict_users = {i: dict_users[i] for i in node_ids if i not in pr_node_ids}
    return pr_dict_users, fr_dict_users

def create_test_data(pr_dict_users, data_train, data_test, no_test_data=None):
    if no_test_data is None:
        no_test_data = len(data_test)

    # Extract priority labels from training data
    pr_labels = []
    for user in pr_dict_users.values():
        pr_labels.extend([data_train[i][1] for i in user])

    # Count occurrences of each label in the priority data
    unique_pr_labels, counts = np.unique(pr_labels, return_counts=True)
    pr_label_proportions = counts / len(pr_labels)

    # Calculate the number of samples per label in the test data
    test_samples_per_label = (pr_label_proportions * no_test_data).astype(int)

    # Extract test data indices based on the calculated proportions
    if isinstance(data_test, torch.utils.data.Subset):
        indices = data_test.indices
        test_labels = data_test.dataset.targets[indices].numpy()
        original_dataset = data_test.dataset
    else:
        test_labels = np.array(data_test.targets)
        original_dataset = data_test
    test_indices = []
    for label, count in zip(unique_pr_labels, test_samples_per_label):
        label_indices = np.where(test_labels == label)[0]
        selected_indices = np.random.choice(label_indices, count, replace=True)
        test_indices.extend(selected_indices)

    # Create test dataset based on the selected indices, using the same type as data_test
#     if isinstance(data_test, torch.utils.data.Subset):
#         test_dataset = type(data_test)(data_test.dataset, test_indices)
#     else:
#         test_dataset = type(data_test)(data_test, test_indices)
    np.random.shuffle(test_indices)
    test_dataset = torch.utils.data.Subset(original_dataset, test_indices)
    return test_dataset


def include_worker_gradient(worker_index, gradient, global_loss, local_loss, error):
    global_loss_np = global_loss
    local_loss_np = local_loss
    if np.abs(global_loss_np - local_loss_np) < error:
        return True
#     elif np.abs(global_loss_np - local_loss_np) < error_ltr:
#         return True
    else:
        return False

def split_data(dataset, data_train, n_nodes):
    if n_nodes > 0:
        if dataset == 'FashionMNIST':
            dict_users = fmnist_partition(data_train, n_nodes)
        elif dataset == 'EMNIST':
            dict_users = emnist_partition(data_train, n_nodes)
        elif dataset == 'CIFAR10':
            dict_users = cifar_partition(data_train, n_nodes)
        elif dataset == 'Shakespeare':
            dict_users = shake_partition(data_train, n_nodes)
        else:
            raise Exception('Unknown dataset name.')
    elif n_nodes == 0:
        return None
    return dict_users
