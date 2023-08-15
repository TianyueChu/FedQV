# FedQV: Leveraging Quadratic Voting in Federated Learning

Implementation of the paper submitted to AAAI 2023.


## Requirements
This code requires the following:
- Python 3.6 or greater
- PyTorch 1.6 or greater
- Torchvision
- Numpy 1.18.5

## Data Preparation

-   Download train and test datasets manually from the given links, or they will use the default links in torchvision.
-   Experiments are run on MNIST, Fashion-MNIST and CIFAR10. [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/) [https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist) [http://www.cs.toronto.edu/∼kriz/cifar.html](http://www.cs.toronto.edu/%E2%88%BCkriz/cifar.html)

## Running the experiments

The baseline experiment trains the model in the conventional way.

-   To train FedQV on CIFAR10  with N=100 under no attacker setting:

```
python main_fed.py --dataset cifar --num_channels 3 --model resnet18 --epochs 100 --gpu 0 --num_users 100 --agg fedqv

```

-   To train the FedQV on MNIST with N=100 under Krum attack with 30% attackers:

```
python main_fed.py --dataset mnist --num_channels 1 --model cnn --epochs 100 --gpu 0 --num_users 100 --agg fedqv --num_attackers 30 --attack_type krum

```

-   To train the FedQV with reputation model on Fashion-MNIST under Scaling attack with 50% attackers:

```
python main_fed.py --dataset fmnist --num_channels 1 --model cnn --epochs 100 --gpu 0 --num_users 100 --agg fedqv --num_attackers 50 --attack_type scaling_attack --rep True

```
-   To train the FedQV with reputation model on MNIST under Backdoor attack with 30% attackers, 30 budget and 0.2 similarity threshold:

```
python main_fed.py --dataset mnist --num_channels 1 --model cnn --epochs 100 --gpu 0 --num_users 100 --agg fedqv --num_attackers 30 --attack_type backdoor --budget 30 --theta 0.2

```

-   To train Multi-Krum with FedQV on Fashion-MNIST under Gaussian attack with 30% attackers:

```
python main_fed.py --dataset fmnist --num_channels 1 --model cnn --epochs 100 --gpu 0 --num_users 100 --agg multi-krum --num_attackers 30 --attack_type gaussian_attack --qv True

```
You can change the default values of other parameters to simulate different conditions. Refer to [options.py](utils/options.py).


## Options

The default values for various parameters parsed to the experiment are given in `options.py`. Details are given some of those parameters:

-   `--dataset:` Default is 'mnist'. Options: 'mnist', 'fmnist', 'cifar'
-   `--iid:` Defaul is False. 
-   `--num_users:` Default is 100.
-   `--seed:` Random Seed. Default is 1.
-   `--model:` Local model. Default is 'cnn'. Options:  'cnn', 'resnet18'
-   `--agg:`Aggregation methods. Default is 'fedavg'. Options: 'fedavg', 'fedqv', 'multi-krum'.
-   `--epochs:` Rounds of training. Default is 100.
-   `--frac:`The fraction of parties. Default is 0.1.

#### FedQV Parameters
-   `--budget:`Voting budget. Default is 30.
-   `--theta:` Similarity threshold. Default is 0.2.
-   `'--rep':` whether use reputation model. Default is False.
-   `--qv:` whether use quadratic voting module. Default is False.

#### Attack Parameters
-   `--num_attackers:` Number of attackers. The default is 0.
-   `--attack_type:` Default is 'lableflip'. Options:  'lableflip', 'backdoor','gaussian_attack','krum_attack','trim_attack','backdoor','scaling_attack'.


#### Environment Initilazaion:
conda activate FedQV_simulation
pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install matplotlib==3.3.4
pip3 install scikit-learn


## References
McMahan, Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In Artificial Intelligence and Statistics (AISTATS), 2017.
Bagdasaryan, Eugene, and Vitaly Shmatikov. "Blind backdoors in deep learning models." 30th USENIX Security Symposium (USENIX Security 21). 2021.
Shaoxiong Ji. (2018, March 30). A PyTorch Implementation of Federated Learning. Zenodo. http://doi.org/10.5281/zenodo.4321561


