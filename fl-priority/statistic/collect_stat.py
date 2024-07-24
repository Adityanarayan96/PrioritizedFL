import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *


class CollectStatistics:
    def __init__(self, dataset, seed, method, results_file_name=None):
        self.dataset = dataset
        self.seed = seed
        self.method = method

        if results_file_name is None:
            results_file_name = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
                                              f'transfer/{dataset}/{seed}/{method}/results.csv')

        self.results_file_name = results_file_name

        with open(results_file_name, 'a') as f:
            f.write(
                'sim_seed,num_iter,training_loss,training_accuracy,test_accuracy\n')
            f.close()

    def collect_stat(self, seed, num_iter, model, train_data_loader, test_data_loader, w_global=None):
        loss_value, train_accuracy = model.accuracy(train_data_loader, w_global, device)
        _, prediction_accuracy = model.accuracy(test_data_loader, w_global, device)

        print("Simulation seed", seed, "Iteration", num_iter, "Loss Value", loss_value,
              "Training accuracy", train_accuracy, "Testing accuracy", prediction_accuracy)

        with open(self.results_file_name, 'a') as f:
            f.write(str(seed) + ',' + str(num_iter) + ',' + str(loss_value) + ','
                    + str(train_accuracy) + ',' + str(prediction_accuracy) + '\n')
            f.close()