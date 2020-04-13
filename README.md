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

## Experiment

We wanted to assess the out-of-distribution generative power of a simple fully connected conditional GAN model. To do so we used the MNIST dataset and train the GAN on digit with thickness between 1-6 pixels. With our framework we experiment the GAN out-of-distribution quality by generating images with thickness between 6-9 pixels. 

Here you can find the main findings of the experiment with qualitative and quantitative assessments. Keep in mind that the **upper images** are **generated images** and the **lower images** are **real images**.

### In distribution quality:

![In Distribution examples : thickness = 4 pxl](https://github.com/RomainGratier/Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/blob/master/references/in-distribution4.png)

### Out-of-distribution quality:

![Out-of-Distribution examples : thickness = 9 pxl](https://github.com/RomainGratier/Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/blob/master/references/out-of-distribution9.png)

![Out-of-Distribution examples : thickness = 8 pxl](https://github.com/RomainGratier/Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/blob/master/references/out-of-distribution8.png)

![Out-of-Distribution examples : thickness = 7 pxl](https://github.com/RomainGratier/Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/blob/master/references/out-of-distribution7.png)

![Out-of-Distribution examples : thickness = 6 pxl](https://github.com/RomainGratier/Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/blob/master/references/out-of-distribution6.png)

### Main Findings:

Table of quantitative results using FID, KID scores and MSE:

| FID (In) | KID(In) | Acc. Thick (In) | FID (Ext)| KID(Ext) | Acc. Thick (Ext) |
| -------- | -------- | -------- | -------- | -------- | -------- |
| Naive generator | 15.0 ± 0.6 | 0.38 ± 0.02 | 1.27 | 28.5 ± 1.7 | **0.91 ± 0.11** | 11.8 |
| w/o IS | 15.0 ± 0.6 | 0.38 ± 0.02 | **0.31 ± 0.20** | 28.5 ± 1.7 | **0.91 ± 0.11** | 8.8 ± 2.6 |
| w/o inference | **13.9 ± 0.7** | **0.33 ± 0.02** | 1.33 | **27.3 ± 1.8** | 0.91 ± 0.13 | 10.7 |
| Full model | **13.9 ± 0.7** | **0.33 ± 0.02** | 0.41 ± 0.39 | **27.3 ± 1.8** | 0.91 ± 0.13 | 4.9 ± 2.7 |



## Getting Started

If you want to use this project on your local machine, start by cloning this repository. Then be sure to have the prerequisites and follow the installation steps.

If you have a google account by simple click on the following links you'll be able to run the experiments on Google Colab 

Experiments :

* The full model is presented here on Colab ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)https://colab.research.google.com/drive/1YY8h7QpnO_MkO_DUpEC86ljhciSfANwK

* The model without IS (importance sampling) reweighting is available here on Colab ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)https://colab.research.google.com/drive/10obGrN3I_gWZSZAY74FQhMomG3qKoRL6


### Prerequisites

```
Python 3.6.9 and older
Torch 1.4.0
```

### Installing

* If you are fine to use it with colab, you can open the link above and the various dependencies should be already satisfied.

* If you want to be able to use the source code and deploy it on your local machine. You will need to clone the following github version (date : 08.04.2020)

```
git clone https://github.com/dccastro/Morpho-MNIST.git
cp -r Morpho-MNIST/morphomnist morphomnist

git clone https://github.com/abdulfatir/gan-metrics-pytorch.git
cp -r gan-metrics-pytorch/models models

git clone https://github.com/RomainGratier/Black-box_Optimization_via_Deep_Generative-Exploratory_Networks.git
cp -r Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/src src
```

## Example of codes

Explain how to run the automated tests for this system

### Create dataset

You can load the morphomnist dataset as a pytorch dataset using the following commands

```
from src.data import MNISTDataset
dataset = MNISTDataset('train', 'thickness')
```

### Load models and their training loop

You can either load the models and play with them following the notebook or just load the training loop to train them.

```
from src.models import Generator, Discriminator, ForwardModel
from src.models import train_forward_model(), train_gan_model()
```

### Inference experiments

You can do inference experiments when the two models are trained

```
from src.inference import monte_carlo_inference
inf_best_5, mean_sample_measure_5, fid_gen_5 = monte_carlo_inference(5, generator, forward, trainset, testset, sample_number = 800)
```

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
