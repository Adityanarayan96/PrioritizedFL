import torch
from torch.utils.data import DataLoader
from config import *
from dataset.dataset import *
from statistic.collect_stat import CollectStatistics
from util.util import *
# from methods.methods import *
import numpy as np
import random
import copy
from model.model import Model
from collections import defaultdict
import sys
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

if device.type != 'cpu':
    torch.cuda.set_device(device)

if __name__ == "__main__":
    
    for seed in simulations:
#         del model
#         del w_global
#         del w_accumulate
        torch.cuda.empty_cache()
        random.seed(seed)
        np.random.seed(seed)  # numpy
        torch.manual_seed(seed)  # cpu
        torch.cuda.manual_seed(seed)  # gpu
        torch.backends.cudnn.deterministic = True  # cudnn
        
        data_train, data_test_w = load_data(dataset, dataset_file_path, 'cpu', config_dict) #Get All Train and Test Data
        dict_users = split_data(dataset, data_train, n_nodes) #Split data to all users

        pr_dict_users, fr_dict_users = allocate_priority(dict_users, pr_nodes) #Seperate them into priority and free clients
        filtered_fr_dict_users = filter_dict_by_labels(fr_dict_users, pr_dict_users, data_train)
          #Create test data based on which clients are 
        #Get a list of trainloaders for the priority clients
#         print(filtered_fr_dict_users)
        pr_train_loader_list = []
        pr_dataiter_list = [] 
        for i,j in zip(pr_dict_users.keys(),range(pr_nodes)):
            pr_train_loader_list.append(
                DataLoader(DatasetSplit(data_train, pr_dict_users[i]), batch_size=batch_size_train, shuffle=True, num_workers=0))
            pr_dataiter_list.append(iter(pr_train_loader_list[j]))
        #Get a list of trainloaders for the free clients
        fr_train_loader_list = []
        fr_dataiter_list = []
        for i,j in zip(fr_dict_users.keys(),range(fr_nodes)):
            fr_train_loader_list.append(
                DataLoader(DatasetSplit(data_train, fr_dict_users[i]), batch_size=batch_size_train, shuffle=True,num_workers=0))
            fr_dataiter_list.append(iter(fr_train_loader_list[j]))  
        
        keys_list = list(filtered_fr_dict_users.keys())
        fr_filter_nodes = len(keys_list)
        # Get the first key from the dictionary
        first_key = next(iter(filtered_fr_dict_users))

# Create a new dictionary with the first key-value pair
        truncated_dict = {first_key: filtered_fr_dict_users[first_key]}
        data_test = create_test_data(truncated_dict, data_train, data_test_w)
#         print(keys_list)
        fr_filtered_train_loader_list = []
        fr_filtered_dataiter_list = []
        for i,j in zip(filtered_fr_dict_users.keys(),range(fr_filter_nodes)):
            fr_filtered_train_loader_list.append(
                DataLoader(DatasetSplit(data_train, filtered_fr_dict_users[i]), batch_size=batch_size_train, shuffle=True,num_workers=0))
            fr_filtered_dataiter_list.append(iter(fr_filtered_train_loader_list[j]))  
        
        merged_indices = []
        for key in pr_dict_users:
            merged_indices.extend(pr_dict_users[key])
        
        data_train_loader = DataLoader(DatasetSplit(data_train, merged_indices), batch_size=batch_size_eval, num_workers=0)
        data_test_loader = DataLoader(data_test, batch_size=batch_size_eval, num_workers=2)
        def sample_minibatch_pr(n):
            try:
                images, labels = pr_dataiter_list[n].next()
                if len(images) < batch_size_train:
                    pr_dataiter_list[n] = iter(pr_train_loader_list[n])
                    images, labels = pr_dataiter_list[n].next()
            except StopIteration:
                pr_dataiter_list[n] = iter(pr_train_loader_list[n])
                images, labels = pr_dataiter_list[n].next()

            return images, labels

        def sample_fullbatch_pr(n):
            images = []
            labels = []
            for i in range(len(pr_train_loader_list[n].dataset)):
                images.append(pr_train_loader_list[n].dataset[i][0])

                l = pr_train_loader_list[n].dataset[i][1]
                if not isinstance(l, torch.Tensor):
                    l = torch.as_tensor(l)
                labels.append(l)

            return torch.stack(images), torch.stack(labels)\
         
        def sample_minibatch_fr(n):
            try:
                images, labels = fr_dataiter_list[n].next()
                if len(images) < batch_size_train:
                    fr_dataiter_list[n] = iter(fr_train_loader_list[n])
                    images, labels = fr_dataiter_list[n].next()
            except StopIteration:
                fr_dataiter_list[n] = iter(fr_train_loader_list[n])
                images, labels = fr_dataiter_list[n].next()

            return images, labels

        def sample_fullbatch_fr(n):
            images = []
            labels = []
            for i in range(len(fr_train_loader_list[n].dataset)):
                images.append(fr_train_loader_list[n].dataset[i][0])

                l = fr_train_loader_list[n].dataset[i][1]
                if not isinstance(l, torch.Tensor):
                    l = torch.as_tensor(l)
                labels.append(l)

            return torch.stack(images), torch.stack(labels)
        def sample_minibatch_fr_filtered(n):
            try:
                images, labels = fr_filtered_dataiter_list[n].next()
                if len(images) < batch_size_train:
                    fr_filtered_dataiter_list[n] = iter(fr_filtered_train_loader_list[n])
                    images, labels = fr_filtered_dataiter_list[n].next()
            except StopIteration:
                fr_filtered_dataiter_list[n] = iter(fr_filtered_train_loader_list[n])
                images, labels = fr_filtered_dataiter_list[n].next()

            return images, labels
        
        for method in methods:
            stat = CollectStatistics(dataset, seed, method, results_file_name=results_file)
            model = Model(seed, step_size_local, model_name=model_name, device=device, flatten_weight=True,
                      pretrained_model_file=load_model_file)
            w_global = model.get_weight()
            num_iter = 0
            last_output = 0
            last_save_latest = 0
            last_save_checkpoint = 0

            while True:
                if method == 'Local':
                    print('seed', seed,'  iteration', num_iter)
                    w_accumulate = None
                    accumulated = 0
                    model.assign_weight(w_global)
                    model.model.train()
#                     fr_fl_node_subset = [i for i in range(0,fr_filter_nodes)]
#                     for n in fr_fl_node_subset:
                    for i in range(0, iters_per_round):
                        images, labels = sample_minibatch_fr_filtered(0)
                        images, labels = images.to(device), labels.to(device)

                        if transform_train is not None:
                            images = transform_train(images)

                        model.optimizer.zero_grad()
                        output = model.model(images)
                        loss = model.loss_fn(output, labels)
                        loss.backward()
                        model.optimizer.step()
                    w_tmp = model.get_weight()  # deepcopy is already included here
                    w_tmp -= w_global

                    if accumulated == 0:  # accumulated weights
                        w_accumulate = w_tmp
                        # Note: w_tmp cannot be used after this
                    else:
                        w_accumulate += w_tmp

                    accumulated += 1
                    w_global += torch.div(w_accumulate, torch.tensor(accumulated).to(device)).view(-1)
#                     _, acc_gl = model.accuracy(data_train_loader, w_global, device)
                    num_iter = num_iter + iters_per_round
                    if num_iter - last_save_latest >=  iters_per_round:
                        print('Saving model')
                        torch.save(model.model.state_dict(), save_model_file)
                        last_save_latest = num_iter

                    if save_checkpoint and num_iter - last_save_checkpoint >= iters_checkpoint:
                        torch.save(model.model.state_dict(), save_model_file + '-checkpoint-sim-' + str(seed) + '-iter-' + str(num_iter))
                        last_save_checkpoint = num_iter

                    if num_iter - last_output >= min_iters_per_eval:
                        stat.collect_stat(seed, num_iter, model, data_train_loader, data_test_loader, w_global)
                        last_output = num_iter

                    if num_iter >= max_iter:
#                         del model
#                         del w_global
#                         del w_accumulate
#                         torch.cuda.empty_cache()
                        break
                else:   
                    print('seed', seed,'  iteration', num_iter)
                    if uniform:
                        pr_node_subset = np.random.choice(pr_nodes, pr_round, replace = False).tolist()
                        fr_node_subset = np.random.choice(fr_nodes, fr_round, replace = False).tolist()
                    else:
                        pr_node_subset = [i for i in range(0,pr_nodes)]
                        fr_node_subset = [i for i in range(0,fr_nodes)]
                    print(pr_node_subset, len(pr_dict_users.keys()), model.learning_rate)
                    w_accumulate = None
                    accumulated = 0
                    for n in pr_node_subset:
                        model.assign_weight(w_global)
                        model.model.train()
                        for i in range(0, iters_per_round):
                            if use_full_batch:
                                images, labels = sample_fullbatch_pr(n)
                            else:
                                images, labels = sample_minibatch_pr(n)

                            images, labels = images.to(device), labels.to(device)

                            if transform_train is not None:
                                images = transform_train(images)

                            model.optimizer.zero_grad()
                            output = model.model(images)
                            loss = model.loss_fn(output, labels)
                            loss.backward()
                            model.optimizer.step()
                        w_tmp = model.get_weight()  # deepcopy is already included here
                        w_tmp -= w_global

                        if accumulated == 0:  # accumulated weights
                            w_accumulate = w_tmp
                            # Note: w_tmp cannot be used after this
                        else:
                            w_accumulate += w_tmp

                        accumulated += 1
                    w_global += torch.div(w_accumulate, torch.tensor(accumulated).to(device)).view(-1)
                    _, acc_gl = model.accuracy(data_train_loader, w_global, device)
                    num_iter = num_iter + iters_per_round
                    if method == 'FedAVG' and num_iter >= iters_warmup:
                        num_iter = num_iter - iters_per_round
                        for n in fr_node_subset:
                            model.assign_weight(w_global)
                            model.model.train()
                            for i in range(0, iters_per_round):
                                if use_full_batch:
                                    images, labels = sample_fullbatch_fr(n)
                                else:
                                    images, labels = sample_minibatch_fr(n)

                                images, labels = images.to(device), labels.to(device)

                                if transform_train is not None:
                                    images = transform_train(images)

                                model.optimizer.zero_grad()
                                output = model.model(images)
                                loss = model.loss_fn(output, labels)
                                loss.backward()
                                model.optimizer.step()
                            w_tmp = model.get_weight()  # deepcopy is already included here
                            w_tmp -= w_global

                            if accumulated == 0:  # accumulated weights
                                w_accumulate = w_tmp
                                # Note: w_tmp cannot be used after this
                            else:
                                w_accumulate += w_tmp

                            accumulated += 1
                        num_iter = num_iter + iters_per_round
                    w_global += torch.div(w_accumulate, torch.tensor(accumulated).to(device)).view(-1)

                    if method == 'FedALIGN' and num_iter >= iters_warmup:
                        num_iter = num_iter - iters_per_round
                        for n in fr_node_subset:
                            model.assign_weight(w_global)
                            model.model.train()
                            images_test, labels_test = sample_minibatch_fr(n) 
                            _, acc = model.accuracy_im(images_test, labels_test, w_global, device)
                            if include_worker_gradient(n, w_tmp, acc, acc_gl, inclusion_threshold, inclusion_threshold_later, num_iter):
                                print('yes')
                                for i in range(0, iters_per_round):
                                    if use_full_batch:
                                        images, labels = sample_fullbatch_fr(n)
                                    else:
                                        images, labels = sample_minibatch_fr(n)

                                    images, labels = images.to(device), labels.to(device)

                                    if transform_train is not None:
                                        images = transform_train(images)

                                    model.optimizer.zero_grad()
                                    output = model.model(images)
                                    loss = model.loss_fn(output, labels)
                                    loss.backward()
                                    model.optimizer.step()
                                w_tmp = model.get_weight()  # deepcopy is already included here
                                w_tmp -= w_global
                                accumulated += 1
                                if accumulated == 0:  # accumulated weights
                                    w_accumulate = w_tmp
                                    # Note: w_tmp cannot be used after this
                                else:
                                    w_accumulate += w_tmp
                        num_iter = num_iter + iters_per_round                                
                    w_global += torch.div(w_accumulate, torch.tensor(accumulated).to(device)).view(-1)

    #                 if num_iter == int(max_iter/2):
    #                     model.learning_rate = 0.1*model.learning_rate
                    if num_iter - last_save_latest >=  iters_per_round:
                        print('Saving model')
                        torch.save(model.model.state_dict(), save_model_file)
                        last_save_latest = num_iter

                    if save_checkpoint and num_iter - last_save_checkpoint >= iters_checkpoint:
                        torch.save(model.model.state_dict(), save_model_file + '-checkpoint-sim-' + str(seed) + '-iter-' + str(num_iter))
                        last_save_checkpoint = num_iter

                    if num_iter - last_output >= min_iters_per_eval:
    #                     stat.collect_stat(seed, num_iter, model, data_train_loader, data_test_loader, w_global)
                        stat.collect_stat(seed, num_iter, model, data_train_loader, data_test_loader, w_global)
                        last_output = num_iter

                    if num_iter >= max_iter:
                        break
            del model
            del w_global
            del w_accumulate
            torch.cuda.empty_cache()
                