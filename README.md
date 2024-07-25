# README

## PFL Experiment Runner

This repository contains a script, `run.py`, which is used to run a series of federated learning experiments with different configurations. The experiments are defined in the script, and the results are saved in structured directories based on the dataset, availability, and priority fraction.

### Requirements

Ensure you have the following installed:
- Python 3.8.19
- Required Python packages (install via `requirements.txt`)

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Adityanarayan96/PrioritizedFL.git
   cd PrioritizedFL
   ```

2. **Install the required Python packages:**

   ```bash
   pip install -r requirements.txt
   ```

### Running Experiments

1. **Define your experiments:**

   Open the `run.py` file and add your experiment configurations to the `experiments` list. Each experiment should be a string containing the necessary parameters.

   Example:
   ```python
   experiments = [
       "-data cifar -availability always -lr-warmup 0.1 -method NONE,FedAVG,FedALIGN -seeds 1,2,3,4,5 -iters-warmup 100 -iters-total 1000 -lr 0.01 -include-gradient-threshold 0.2 -include-gradient-threshold_later 0.2 -total-clients 60 -priorityfrac 0.07 ",
       "-data emnist -availability uniform -lr-warmup 0.1 -method NONE,FedAVG,FedALIGN -seeds 1,2,3,4,5 -iters-warmup 100 -iters-total 1000 -lr 0.1 -include-gradient-threshold 0.4 -include-gradient-threshold_later 0.2 -total-clients 60 -priorityfrac 0.07 ",
       # Add more experiments here...
   ]
   ```

2. **Run the experiments:**

   Execute the `run.py` script to run the defined experiments. This will execute `main.py` with the specified arguments for each experiment and save the results in structured directories.

   ```bash
   python run.py
   ```

### Directory Structure

The results of the experiments will be saved in the `transfer` directory, organized by dataset, availability, and priority fraction. For example:

```
transfer/
└── cifar/
    └── always/
        └── 0.07/
            └── results.csv
```

### Customization

You can customize the following parts of the `run.py` script:
- **Experiments list:** Add or modify experiments in the `experiments` list.
- **Command base:** The `command_base` variable defines the base command to run `main.py` with the specified arguments and output path.
- **Output placeholders:** You can change the placeholder used in the output filename to uniquely identify results.

### Contact

For any questions or issues, please contact Aditya Narayan Ravi at anravi2@illinois.edu.

---

This README provides a basic guide on how to use the script to run federated learning experiments. Customize the experiments as needed and follow the structure to ensure your results are saved correctly.
