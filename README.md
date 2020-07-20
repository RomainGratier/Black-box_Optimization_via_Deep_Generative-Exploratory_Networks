Black-box Optimization via Deep Generative-Exploratory Networks
==============================

In this project, we assess a new framework called UMIN on a data-driven optimization problem. Such a problem happens recurrently in real life and can quickly become difficult to model when the input has a high dimensionality as images for instance. From the architecture of aircraft to the design of proteins, a great number of different techniques have already been explored. Based on former solutions, this work introduces a brand new Bayesian approach that updates previous frameworks. Former model architectures use generative adversarial networks on one side and a forward model on the other side to improve the accuracy of the results. However, employing a Bayesian forward network allows us to leverage its uncertainty estimates to enhance the accuracy of the results and also to reduce unrealistic samples output by the generator. By creating new experiments on a modern MNIST dataset and by reproducing former works taken as baseline, we show that the framework introduces in this work outperforms the previous method. The whole code is available at the following url: [PurpleBooth](https://github.com/RomainGratier/Black-box_Optimization_via_Deep_Generative-Exploratory_Networks).

We used well known metrics from the generative model community as the **FID**(Fréchet Inception Distance), **KID**(Kernel Inception Distance) and **MSE**(mean squared error) scores to justify our results.

------------

    ├── README.md                           <- The top-level README for developers using this project.
    ├── data                                <- Data collected from MNIST 
    │
    ├── models                              <- incepetion models to compute fid/kid scores
    │
    ├── MNIST_generation                    <- Scripts to augment original MNIST dataset.
    │
    ├── fig                                 <- Distribution figures from augmented MNIST dataset.
    │
    ├── requirements.txt                    <- The requirements file for reproducing the analysis environment, e.g.
    │                                           generated with `pip freeze > requirements.txt`
    │
    ├── setup.py                            <- makes project pip installable (pip install -e .) so src can be imported
    ├── utils.py                            <- Plot functions repository
    ├── UMIN                                <- Source code for use in this project.
    │   ├── __init__.py                     <- Makes src a Python module
    │   │
    │   ├── data                            <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── metrics                         <- Source of different metrics  
    │   │   └── metrics.py
    │   │
    │   ├── generative_model                <- Generators training scripts
    │   │   └── train_model.py
    |   |
    │   ├── forward_model                   <- forward models training scripts
    │   │   ├── inference_func_bayesian.py
    |   |   |
    |   |   └── inference_func_frequentist.py
    |   |
    │   ├── uncertainty_policy              <- Uncertainty selection function
    │   │   └── policy.py
    │   │
    │   └── Inference                       <- Script to run the model inference
    │       └── inference.py
    │
    └── tox.ini                             <- tox file with settings for running tox; see tox.testrun.org


--------

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us. 

## Authors

* **Romain Gratier de Saint-Louis** [PurpleBooth](https://github.com/RomainGratier)
* **Yuejiang Liu** - *VITA Lab EPFL* - [PurpleBooth](https://github.com/YuejiangLIU)
* **Alexandre Alahi** - *VITA Lab EPFL* 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Special thanks the to writers of the paper available on arxiv : https://arxiv.org/pdf/1912.13464.pdf. Their contribution motivates this project.

An other special thanks the to writers of the paper available on arxiv ( https://arxiv.org/pdf/1901.02731.pdf ) and their github repository to allow us to use accurate Bayesian neural networks.

Bravo to the work done on https://github.com/dccastro/Morpho-MNIST that allows us to find an interesting dataset and established groundtruth measurements for our generated data

Bravo also to https://github.com/eriklindernoren/PyTorch-GAN to share the pytorch implementation of usefull GANs

And Bravo to https://github.com/abdulfatir/gan-metrics-pytorch.git that implemented a great FID score for MNIST dataset
