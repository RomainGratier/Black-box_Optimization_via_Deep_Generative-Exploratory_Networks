Black-box Optimization via Deep Generative-Exploratory Networks
==============================

Deep neural networks have garnered tremendous excitement in recent years thanks to their superior learning capacity in the presence of abundant data resources. However, collecting an exhaustive dataset covering all possible scenarios is often slow, expensive, and even impractical. 

**The goal** of this project is to devise a new **learning framework** that can learn from a finite dataset and noisy feedback of data properties **to discover novel samples** of particular interest. We used generative adversarial networks to search through the dataset for new sample and a forward model to decide what sample is a potential candidate.

We used well known metrics from the generative model community as the **FID**(Fréchet Inception Distance), **KID**(Kernel Inception Distance) and **MSE**(mean squared error) scores to justify our results.

------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Data collected from MNIST 
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── metrics        <- Source of different metrics  
    │   │   └── metrics.py
    │   │
    │   ├── models         <- Scripts to train models and then use load models and trained them
    │   │   │                 
    │   │   ├── models.py
    │   │   └── train_model.py
    │   │
    │   └── Inference      <- Script to run the model inference
    │       └── inference.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us. 

## Authors

* **Romain Gratier de Saint-Louis** [PurpleBooth](https://github.com/RomainGratier)
* **Yuejiang Liu** - *VITA Lab EPFL* - [PurpleBooth](https://github.com/YuejiangLIU)
* **Alexandre Alahi** - *VITA Lab EPFL* 

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Special thanks the to writers of the paper available on arxiv : https://arxiv.org/pdf/1912.13464.pdf. Their contribution motivates this project.

Bravo to the work done on https://github.com/dccastro/Morpho-MNIST that allows us to find an interesting dataset and established groundtruth measurements for our generated data

Bravo also to https://github.com/eriklindernoren/PyTorch-GAN to share the pytorch implementation of usefull GANs

And Bravo to https://github.com/abdulfatir/gan-metrics-pytorch.git that implemented a great FID score for MNIST dataset
