import os
import shutil

# Define a list of experiments to run
experiments = [
    # Add your experiments here, for example:
    "-data fmnist -availability always -lr-warmup 0.1 -method NONE,FedAVG,FedALIGN -seeds 1,2,3,4,5 -iters-warmup 100 -iters-total 1000 -lr 0.1 -include-gradient-threshold 0.2 -include-gradient-threshold_later 0.2 -total-clients 60 -priorityfrac 0.034 ",
    "-data emnist -availability always -lr-warmup 0.1 -method NONE,FedAVG,FedALIGN -seeds 1,2,3,4,5 -iters-warmup 100 -iters-total 1000 -lr 0.1 -include-gradient-threshold 0.2 -include-gradient-threshold_later 0.2 -total-clients 60 -priorityfrac 0.034 ",
    "-data cifar10 -availability always -lr-warmup 0.1 -method NONE,FedAVG,FedALIGN -seeds 1,2,3,4,5 -iters-warmup 100 -iters-total 1000 -lr 0.01 -include-gradient-threshold 0.2 -include-gradient-threshold_later 0.2 -total-clients 60 -priorityfrac 0.034 ",
#     "-data emnist -availability uniform -lr-warmup 0.1 -method None,FedAVG,FedALIGN -seeds 1,2,3,4,5 -iters-warmup 500 -iters-total 5000 -lr 0.1 -include-gradient-threshold 0.2 -include-gradient-threshold_later 0.2 -total-clients 100 -priorityfrac 0.2 ",
#     "-data cifar10 -availability uniform -lr-warmup 0.1 -method None,FedAVG,FedALIGN -seeds 1,2,3,4,5 -iters-warmup 500 -iters-total 5000 -lr 0.01 -include-gradient-threshold 0.2 -include-gradient-threshold_later 0.2 -total-clients 100 -priorityfrac 0.2 ",    
    # ...
]

# Define the base command to run the experiments
command_base = "python3 main.py {args} -out transfer/{dataset}/{availability}/{priorityfrac}/{placeholder}.csv"

for i, experiment in enumerate(experiments):
    # Extract the dataset name from the experiment string
    data_start = experiment.find("-data ") + 6
    data_end = experiment.find(" ", data_start)
    dataset = experiment[data_start:data_end]

    # Extract the availability from the experiment string
    availability_start = experiment.find("-availability ") + 14
    availability_end = experiment.find(" ", availability_start)
    availability = experiment[availability_start:availability_end]

    # Extract the priorityfrac from the experiment string
    priorityfrac_start = experiment.find("-priorityfrac ") + 14
    priorityfrac_end = experiment.find(" ", priorityfrac_start)
    priorityfrac = experiment[priorityfrac_start:priorityfrac_end]

    if os.path.exists(f"transfer/{dataset}/{availability}/{priorityfrac}/") == False:
#         shutil.rmtree(f"transfer/{dataset}")  # Delete the contents of the directory
        os.makedirs(f"transfer/{dataset}/{availability}/{priorityfrac}/")

    # Build and run the command with the dataset, method, and unique identifier
    command = command_base.format(args=experiment, dataset=dataset, availability=availability, priorityfrac=priorityfrac, placeholder="results")
    os.system(command)