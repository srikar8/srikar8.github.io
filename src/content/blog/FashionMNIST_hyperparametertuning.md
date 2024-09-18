---
title: Optuna HyperparameterOptimization
description: Fashion MNIST Classification Hyperparameter Optimization using Optuna
heroImage: /Screenshot 2024-09-18 at 2.05.55 PM.png
pubDate: 2024-09-18T19:14:02.035Z
---
### Hyperparameters tuning using Optuna.


Optuna is an open-source hyperparameter optimization framework designed to automate the search for optimal hyperparameters in machine learning models. It helps data scientists and machine learning engineers improve their models by finding the best set of parameters in a more efficient and systematic way.

### List of Hyperparameters
1. Number of layers (n_layers)
2. Number of units per layer (n_units_l{i})
3. Dropout rate (dropout_l{i})
4. Optimizer type (Adam, RMSprop, or SGD)
5. Learning rate (lr)


```python
"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.

"""

import os

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

```

### Basic parameters for a machine learning model:

1. The device (GPU/CPU),
2. Batch size (128),
3. Number of classes (10),
4. Current working directory,
5. Number of epochs (10),
6. Number of training (3840) and validation examples (1280).


```python
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
BATCHSIZE = 128
CLASSES = 10
DIR = os.getcwd()
EPOCHS = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10
```

### Model Definition (define_model):
1. This function creates a neural network architecture based on the trial parameters. The number of layers units per layer, and dropout rate for each layer are sampled as hyperparameters.
2. n_layers is selected between 1 and 3.
3. Each layer's hidden units (n_units_l{i}) vary between 4 and 128.
4. Dropout rate (dropout_l{i}) is chosen between 0.2 and 0.5.


```python
def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []

    in_features = 28 * 28
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)

```

### Dataset Loading (get_mnist):
The FashionMNIST dataset is loaded and transformed into tensor format. The dataset is divided into training and validation sets using DataLoader.

### Labels :

0 T-shirt/top
1 Trouser
2 Pullover
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle boot


```python
def get_mnist():
    # Load FashionMNIST dataset.
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(DIR, train=True, download=True, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(DIR, train=False, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )

    return train_loader, valid_loader
```


```python
import matplotlib.pyplot as plt

train_loader, valid_loader = get_mnist()

# Display a few images from the train_loader
def show_images(loader):
    # Get a batch of training data
    data_iter = iter(loader)  # Create an iterator
    images, labels = next(data_iter)  # Get the next batch of images

    # Plot the images
    fig, axes = plt.subplots(1, 6, figsize=(12, 12))
    for i in range(6):  # Display first 6 images
        ax = axes[i]
        ax.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis('off')

    plt.show()

# Call the function to display images
show_images(train_loader)
```


    
![png](/public/FashionMNIST_hyperparametertuning_files/FashionMNIST_hyperparametertuning_8_0.png)
    


### Objective Function (objective):
1. The objective function generates a model and optimizer for each trial, trains the model, and evaluates it on a subset of validation data.
2. Optimizer: The optimizer type (Adam, RMSprop, or SGD) and learning rate are optimized.
3. Training: The model is trained over a fixed number of epochs (10), with each epoch iterating over mini-batches of training data.
4. Validation: After each epoch, the validation accuracy is computed.
5. If the model isn't performing well by certain checkpoints, the trial can be pruned to save time (early stopping).


```python
def objective(trial):
    # Generate the model.
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the FashionMNIST dataset.
    train_loader, valid_loader = get_mnist()

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy
```

### Main Execution:
1. An Optuna study is created with the objective to maximize validation accuracy.
2. The study.optimize() method runs the optimization for 100 trials or for a timeout of 600 seconds, whichever comes first.
3. After completion, statistics and the best trial's parameters are printed.


```python
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
```

    [I 2024-09-18 13:30:40,032] A new study created in memory with name: no-name-76521e9d-2fdc-40fc-9663-c04a9cc6a54f
    [I 2024-09-18 13:30:44,988] Trial 0 finished with value: 0.7765625 and parameters: {'n_layers': 1, 'n_units_l0': 57, 'dropout_l0': 0.27938910253310106, 'optimizer': 'RMSprop', 'lr': 0.0003494675062008697}. Best is trial 0 with value: 0.7765625.
    [I 2024-09-18 13:30:49,642] Trial 1 finished with value: 0.3234375 and parameters: {'n_layers': 2, 'n_units_l0': 124, 'dropout_l0': 0.3731705227477192, 'n_units_l1': 29, 'dropout_l1': 0.22535103447479576, 'optimizer': 'RMSprop', 'lr': 0.02921719024101408}. Best is trial 0 with value: 0.7765625.
    [I 2024-09-18 13:30:54,105] Trial 2 finished with value: 0.14765625 and parameters: {'n_layers': 1, 'n_units_l0': 94, 'dropout_l0': 0.24064426334739136, 'optimizer': 'SGD', 'lr': 0.00017211314101429242}. Best is trial 0 with value: 0.7765625.
    [I 2024-09-18 13:30:58,613] Trial 3 finished with value: 0.58359375 and parameters: {'n_layers': 1, 'n_units_l0': 99, 'dropout_l0': 0.4705796401526111, 'optimizer': 'RMSprop', 'lr': 1.2829427252560124e-05}. Best is trial 0 with value: 0.7765625.
    [I 2024-09-18 13:31:03,332] Trial 4 finished with value: 0.25859375 and parameters: {'n_layers': 3, 'n_units_l0': 88, 'dropout_l0': 0.4775656409015328, 'n_units_l1': 54, 'dropout_l1': 0.35919328072167134, 'n_units_l2': 16, 'dropout_l2': 0.36794623987167147, 'optimizer': 'RMSprop', 'lr': 2.2958167824378915e-05}. Best is trial 0 with value: 0.7765625.
    [I 2024-09-18 13:31:07,838] Trial 5 finished with value: 0.63359375 and parameters: {'n_layers': 1, 'n_units_l0': 32, 'dropout_l0': 0.22207651616823665, 'optimizer': 'RMSprop', 'lr': 5.5731692385700076e-05}. Best is trial 0 with value: 0.7765625.
    [I 2024-09-18 13:31:12,513] Trial 6 finished with value: 0.63046875 and parameters: {'n_layers': 3, 'n_units_l0': 70, 'dropout_l0': 0.45637067452432833, 'n_units_l1': 58, 'dropout_l1': 0.20860803968244387, 'n_units_l2': 53, 'dropout_l2': 0.46706573294044695, 'optimizer': 'RMSprop', 'lr': 0.0002626349427222672}. Best is trial 0 with value: 0.7765625.
    [I 2024-09-18 13:31:13,038] Trial 7 pruned. 
    [I 2024-09-18 13:31:17,713] Trial 8 pruned. 
    [I 2024-09-18 13:31:18,239] Trial 9 pruned. 
    [I 2024-09-18 13:31:22,802] Trial 10 finished with value: 0.80859375 and parameters: {'n_layers': 1, 'n_units_l0': 65, 'dropout_l0': 0.3013490333350582, 'optimizer': 'Adam', 'lr': 0.003435526197544358}. Best is trial 10 with value: 0.80859375.
    [I 2024-09-18 13:31:27,323] Trial 11 finished with value: 0.815625 and parameters: {'n_layers': 1, 'n_units_l0': 57, 'dropout_l0': 0.30820081456825926, 'optimizer': 'Adam', 'lr': 0.0034628279828409986}. Best is trial 11 with value: 0.815625.
    [I 2024-09-18 13:31:31,871] Trial 12 finished with value: 0.81796875 and parameters: {'n_layers': 2, 'n_units_l0': 70, 'dropout_l0': 0.32119844970036754, 'n_units_l1': 127, 'dropout_l1': 0.48836374820536677, 'optimizer': 'Adam', 'lr': 0.003714612981715545}. Best is trial 12 with value: 0.81796875.
    [I 2024-09-18 13:31:36,487] Trial 13 finished with value: 0.80390625 and parameters: {'n_layers': 2, 'n_units_l0': 77, 'dropout_l0': 0.40812082845285974, 'n_units_l1': 127, 'dropout_l1': 0.4938834911557537, 'optimizer': 'Adam', 'lr': 0.0026616842598306567}. Best is trial 12 with value: 0.81796875.
    [I 2024-09-18 13:31:41,121] Trial 14 finished with value: 0.7984375 and parameters: {'n_layers': 2, 'n_units_l0': 47, 'dropout_l0': 0.32745180094616005, 'n_units_l1': 96, 'dropout_l1': 0.48559907196360946, 'optimizer': 'Adam', 'lr': 0.006509307404167071}. Best is trial 12 with value: 0.81796875.
    [I 2024-09-18 13:31:41,661] Trial 15 pruned. 
    [I 2024-09-18 13:31:46,186] Trial 16 finished with value: 0.78984375 and parameters: {'n_layers': 1, 'n_units_l0': 45, 'dropout_l0': 0.35546335052328665, 'optimizer': 'Adam', 'lr': 0.0011081791882109747}. Best is trial 12 with value: 0.81796875.
    [I 2024-09-18 13:31:50,797] Trial 17 finished with value: 0.7765625 and parameters: {'n_layers': 2, 'n_units_l0': 81, 'dropout_l0': 0.426220677185332, 'n_units_l1': 78, 'dropout_l1': 0.30613925259981706, 'optimizer': 'Adam', 'lr': 0.010717523470493137}. Best is trial 12 with value: 0.81796875.
    [I 2024-09-18 13:31:52,260] Trial 18 pruned. 
    [I 2024-09-18 13:31:52,777] Trial 19 pruned. 
    [I 2024-09-18 13:31:53,338] Trial 20 pruned. 
    [I 2024-09-18 13:31:57,888] Trial 21 finished with value: 0.81796875 and parameters: {'n_layers': 1, 'n_units_l0': 68, 'dropout_l0': 0.30821141590747453, 'optimizer': 'Adam', 'lr': 0.003306860577328637}. Best is trial 12 with value: 0.81796875.
    [I 2024-09-18 13:31:58,849] Trial 22 pruned. 
    [I 2024-09-18 13:32:03,378] Trial 23 finished with value: 0.821875 and parameters: {'n_layers': 1, 'n_units_l0': 55, 'dropout_l0': 0.33599015655295755, 'optimizer': 'Adam', 'lr': 0.004312163354624081}. Best is trial 23 with value: 0.821875.
    [I 2024-09-18 13:32:07,955] Trial 24 finished with value: 0.821875 and parameters: {'n_layers': 1, 'n_units_l0': 69, 'dropout_l0': 0.3444581857293767, 'optimizer': 'Adam', 'lr': 0.0271028852202935}. Best is trial 23 with value: 0.821875.
    [I 2024-09-18 13:32:08,481] Trial 25 pruned. 
    [I 2024-09-18 13:32:09,015] Trial 26 pruned. 
    [I 2024-09-18 13:32:13,527] Trial 27 finished with value: 0.81796875 and parameters: {'n_layers': 1, 'n_units_l0': 87, 'dropout_l0': 0.4087903136474955, 'optimizer': 'Adam', 'lr': 0.007816711061423079}. Best is trial 23 with value: 0.821875.
    [I 2024-09-18 13:32:16,321] Trial 28 pruned. 
    [I 2024-09-18 13:32:16,842] Trial 29 pruned. 
    [I 2024-09-18 13:32:17,361] Trial 30 pruned. 
    [I 2024-09-18 13:32:21,927] Trial 31 finished with value: 0.81171875 and parameters: {'n_layers': 1, 'n_units_l0': 66, 'dropout_l0': 0.32640237508635145, 'optimizer': 'Adam', 'lr': 0.005157251647401679}. Best is trial 23 with value: 0.821875.
    [I 2024-09-18 13:32:26,458] Trial 32 finished with value: 0.809375 and parameters: {'n_layers': 1, 'n_units_l0': 51, 'dropout_l0': 0.34435427616317393, 'optimizer': 'Adam', 'lr': 0.019585168424472426}. Best is trial 23 with value: 0.821875.
    [I 2024-09-18 13:32:27,427] Trial 33 pruned. 
    [I 2024-09-18 13:32:27,956] Trial 34 pruned. 
    [I 2024-09-18 13:32:28,489] Trial 35 pruned. 
    [I 2024-09-18 13:32:28,999] Trial 36 pruned. 
    [I 2024-09-18 13:32:33,570] Trial 37 finished with value: 0.82890625 and parameters: {'n_layers': 1, 'n_units_l0': 99, 'dropout_l0': 0.34063998290025577, 'optimizer': 'RMSprop', 'lr': 0.005252715792962132}. Best is trial 37 with value: 0.82890625.
    [I 2024-09-18 13:32:34,513] Trial 38 pruned. 
    [I 2024-09-18 13:32:35,050] Trial 39 pruned. 
    [I 2024-09-18 13:32:35,994] Trial 40 pruned. 
    [I 2024-09-18 13:32:37,785] Trial 41 pruned. 
    [I 2024-09-18 13:32:39,158] Trial 42 pruned. 
    [I 2024-09-18 13:32:40,102] Trial 43 pruned. 
    [I 2024-09-18 13:32:40,612] Trial 44 pruned. 
    [I 2024-09-18 13:32:41,155] Trial 45 pruned. 
    [I 2024-09-18 13:32:42,118] Trial 46 pruned. 
    [I 2024-09-18 13:32:42,679] Trial 47 pruned. 
    [I 2024-09-18 13:32:43,221] Trial 48 pruned. 
    [I 2024-09-18 13:32:47,816] Trial 49 finished with value: 0.7984375 and parameters: {'n_layers': 1, 'n_units_l0': 119, 'dropout_l0': 0.31491363165244846, 'optimizer': 'RMSprop', 'lr': 0.0030715085736871474}. Best is trial 37 with value: 0.82890625.
    [I 2024-09-18 13:32:48,355] Trial 50 pruned. 
    [I 2024-09-18 13:32:52,953] Trial 51 finished with value: 0.83359375 and parameters: {'n_layers': 1, 'n_units_l0': 85, 'dropout_l0': 0.4155710954764109, 'optimizer': 'Adam', 'lr': 0.008886576169053525}. Best is trial 51 with value: 0.83359375.
    [I 2024-09-18 13:32:57,495] Trial 52 finished with value: 0.83671875 and parameters: {'n_layers': 1, 'n_units_l0': 83, 'dropout_l0': 0.3570572360302212, 'optimizer': 'Adam', 'lr': 0.004653787226092292}. Best is trial 52 with value: 0.83671875.
    [I 2024-09-18 13:33:02,032] Trial 53 finished with value: 0.81484375 and parameters: {'n_layers': 1, 'n_units_l0': 83, 'dropout_l0': 0.4236600083914652, 'optimizer': 'Adam', 'lr': 0.012046125984131718}. Best is trial 52 with value: 0.83671875.
    [I 2024-09-18 13:33:02,558] Trial 54 pruned. 
    [I 2024-09-18 13:33:07,098] Trial 55 finished with value: 0.80625 and parameters: {'n_layers': 1, 'n_units_l0': 95, 'dropout_l0': 0.35679631959801084, 'optimizer': 'Adam', 'lr': 0.008242744976217097}. Best is trial 52 with value: 0.83671875.
    [I 2024-09-18 13:33:07,608] Trial 56 pruned. 
    [I 2024-09-18 13:33:08,152] Trial 57 pruned. 
    [I 2024-09-18 13:33:08,688] Trial 58 pruned. 
    [I 2024-09-18 13:33:10,107] Trial 59 pruned. 
    [I 2024-09-18 13:33:14,675] Trial 60 finished with value: 0.8328125 and parameters: {'n_layers': 1, 'n_units_l0': 96, 'dropout_l0': 0.32616889124904236, 'optimizer': 'Adam', 'lr': 0.00568042762525041}. Best is trial 52 with value: 0.83671875.
    [I 2024-09-18 13:33:15,214] Trial 61 pruned. 
    [I 2024-09-18 13:33:19,747] Trial 62 finished with value: 0.83671875 and parameters: {'n_layers': 1, 'n_units_l0': 94, 'dropout_l0': 0.3446224495384511, 'optimizer': 'Adam', 'lr': 0.004955040382994126}. Best is trial 52 with value: 0.83671875.
    [I 2024-09-18 13:33:24,294] Trial 63 finished with value: 0.82265625 and parameters: {'n_layers': 1, 'n_units_l0': 92, 'dropout_l0': 0.33418609792483395, 'optimizer': 'Adam', 'lr': 0.005695434410228134}. Best is trial 52 with value: 0.83671875.
    [I 2024-09-18 13:33:28,816] Trial 64 finished with value: 0.8375 and parameters: {'n_layers': 1, 'n_units_l0': 95, 'dropout_l0': 0.3328197996457107, 'optimizer': 'Adam', 'lr': 0.005524560899865061}. Best is trial 64 with value: 0.8375.
    [I 2024-09-18 13:33:33,335] Trial 65 finished with value: 0.82734375 and parameters: {'n_layers': 1, 'n_units_l0': 94, 'dropout_l0': 0.29415736345279186, 'optimizer': 'Adam', 'lr': 0.006564339390554939}. Best is trial 64 with value: 0.8375.
    [I 2024-09-18 13:33:37,883] Trial 66 finished with value: 0.81328125 and parameters: {'n_layers': 1, 'n_units_l0': 105, 'dropout_l0': 0.2694759221716994, 'optimizer': 'Adam', 'lr': 0.008968441342266835}. Best is trial 64 with value: 0.8375.
    [I 2024-09-18 13:33:42,293] Trial 67 finished with value: 0.79921875 and parameters: {'n_layers': 1, 'n_units_l0': 96, 'dropout_l0': 0.2993226591228219, 'optimizer': 'Adam', 'lr': 0.006138267594912107}. Best is trial 64 with value: 0.8375.
    [I 2024-09-18 13:33:42,808] Trial 68 pruned. 
    [I 2024-09-18 13:33:43,318] Trial 69 pruned. 
    [I 2024-09-18 13:33:44,274] Trial 70 pruned. 
    [I 2024-09-18 13:33:48,733] Trial 71 finished with value: 0.8421875 and parameters: {'n_layers': 1, 'n_units_l0': 92, 'dropout_l0': 0.3619160535009199, 'optimizer': 'Adam', 'lr': 0.0060297534701846}. Best is trial 71 with value: 0.8421875.
    [I 2024-09-18 13:33:53,246] Trial 72 finished with value: 0.828125 and parameters: {'n_layers': 1, 'n_units_l0': 96, 'dropout_l0': 0.3751336066699924, 'optimizer': 'Adam', 'lr': 0.004302047787163249}. Best is trial 71 with value: 0.8421875.
    [I 2024-09-18 13:33:57,755] Trial 73 finished with value: 0.8234375 and parameters: {'n_layers': 1, 'n_units_l0': 97, 'dropout_l0': 0.3640713214694381, 'optimizer': 'Adam', 'lr': 0.010674591667769559}. Best is trial 71 with value: 0.8421875.
    [I 2024-09-18 13:34:02,317] Trial 74 finished with value: 0.8046875 and parameters: {'n_layers': 1, 'n_units_l0': 90, 'dropout_l0': 0.3735054037600943, 'optimizer': 'Adam', 'lr': 0.004148735616384576}. Best is trial 71 with value: 0.8421875.
    [I 2024-09-18 13:34:03,289] Trial 75 pruned. 
    [I 2024-09-18 13:34:05,163] Trial 76 pruned. 
    [I 2024-09-18 13:34:08,846] Trial 77 pruned. 
    [I 2024-09-18 13:34:13,448] Trial 78 finished with value: 0.81875 and parameters: {'n_layers': 1, 'n_units_l0': 92, 'dropout_l0': 0.38890955219319423, 'optimizer': 'Adam', 'lr': 0.005460999292580465}. Best is trial 71 with value: 0.8421875.
    [I 2024-09-18 13:34:14,875] Trial 79 pruned. 
    [I 2024-09-18 13:34:16,297] Trial 80 pruned. 
    [I 2024-09-18 13:34:16,830] Trial 81 pruned. 
    [I 2024-09-18 13:34:17,813] Trial 82 pruned. 
    [I 2024-09-18 13:34:18,789] Trial 83 pruned. 
    [I 2024-09-18 13:34:21,588] Trial 84 pruned. 
    [I 2024-09-18 13:34:26,153] Trial 85 finished with value: 0.81953125 and parameters: {'n_layers': 1, 'n_units_l0': 83, 'dropout_l0': 0.35093813650187655, 'optimizer': 'Adam', 'lr': 0.006141017099906333}. Best is trial 71 with value: 0.8421875.
    [I 2024-09-18 13:34:30,667] Trial 86 finished with value: 0.83515625 and parameters: {'n_layers': 1, 'n_units_l0': 109, 'dropout_l0': 0.3207539523920586, 'optimizer': 'Adam', 'lr': 0.0028963236681893998}. Best is trial 71 with value: 0.8421875.
    [I 2024-09-18 13:34:31,184] Trial 87 pruned. 
    [I 2024-09-18 13:34:31,708] Trial 88 pruned. 
    [I 2024-09-18 13:34:32,230] Trial 89 pruned. 
    [I 2024-09-18 13:34:33,187] Trial 90 pruned. 
    [I 2024-09-18 13:34:37,722] Trial 91 finished with value: 0.81015625 and parameters: {'n_layers': 1, 'n_units_l0': 89, 'dropout_l0': 0.3070073412453505, 'optimizer': 'Adam', 'lr': 0.006674855198673993}. Best is trial 71 with value: 0.8421875.
    [I 2024-09-18 13:34:42,213] Trial 92 finished with value: 0.82265625 and parameters: {'n_layers': 1, 'n_units_l0': 98, 'dropout_l0': 0.3232240708936975, 'optimizer': 'Adam', 'lr': 0.003442935255866055}. Best is trial 71 with value: 0.8421875.
    [I 2024-09-18 13:34:42,728] Trial 93 pruned. 
    [I 2024-09-18 13:34:47,146] Trial 94 finished with value: 0.8390625 and parameters: {'n_layers': 1, 'n_units_l0': 79, 'dropout_l0': 0.3525212983907694, 'optimizer': 'Adam', 'lr': 0.009123991904563575}. Best is trial 71 with value: 0.8421875.
    [I 2024-09-18 13:34:47,664] Trial 95 pruned. 
    [I 2024-09-18 13:34:48,164] Trial 96 pruned. 
    [I 2024-09-18 13:34:49,103] Trial 97 pruned. 
    [I 2024-09-18 13:34:49,634] Trial 98 pruned. 
    [I 2024-09-18 13:34:54,127] Trial 99 finished with value: 0.81953125 and parameters: {'n_layers': 1, 'n_units_l0': 102, 'dropout_l0': 0.4072444474766436, 'optimizer': 'Adam', 'lr': 0.004967338854466597}. Best is trial 71 with value: 0.8421875.


    Study statistics: 
      Number of finished trials:  100
      Number of pruned trials:  56
      Number of complete trials:  44
    Best trial:
      Value:  0.8421875
      Params: 
        n_layers: 1
        n_units_l0: 92
        dropout_l0: 0.3619160535009199
        optimizer: Adam
        lr: 0.0060297534701846


### Final Result:

1. Study statistics: 
  Number of finished trials:  100
  Number of pruned trials:  56
  Number of complete trials:  44
2. Best trial:
  Value:  0.8421875
  Params: 
    n_layers: 1
    n_units_l0: 92
    dropout_l0: 0.3619160535009199
    optimizer: Adam
    lr: 0.0060297534701846

##### Referred from https://optuna.org/#code_examples
