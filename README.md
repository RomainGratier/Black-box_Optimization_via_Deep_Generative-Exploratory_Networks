Black-box Optimization via Deep Generative-Exploratory Networks
==============================

In this project, we assess a new framework called UMIN on a data-driven optimization problem. Such a problem happens recurrently in real life and can quickly become difficult to model when the input has a high dimensionality as images for instance. From the architecture of aircraft to the design of proteins, a great number of different techniques have already been explored. Based on former solutions, this work introduces a brand new Bayesian approach that updates previous frameworks. Former model architectures use generative adversarial networks on one side and a forward model on the other side to improve the accuracy of the results. However, employing a Bayesian forward network allows us to leverage its uncertainty estimates to enhance the accuracy of the results and also to reduce unrealistic samples output by the generator. By creating new experiments on a modern MNIST dataset and by reproducing former works taken as baseline, we show that the framework introduces in this work outperforms the previous method.

We used well-known metrics from the generative model community as the **FID**(Fréchet Inception Distance), **KID**(Kernel Inception Distance), and **MSE**(mean squared error) scores to justify our results.

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
    ├── src                                 <- Source code for use in this project.
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

## Getting Started

If you have a google account by simple click on the following links you'll be able to run the experiments on Google Colab 

Experiments :

* The maximization experiment is presented here on Colab ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)https://colab.research.google.com/drive/1Gg8Iz8SEzYr6NcEqcZvwJgkh2Mm1UYqQ?usp=sharing

* The minimization experiment is available here on Colab ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)https://colab.research.google.com/drive/1LbHNNQ3IUFSMcxD1pa7pIkBW49RuI_vi?usp=sharing


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

## Dataset
A new augmented MNIST dataset was used to assess the framework quality. This dataset is build using the **MNIST_generation** scripts but one can also use the dataset created and available in the folder **data**. This dataset allows us to have an image of a digit with different characteristics labeled. The figure below shows those features and their distribution within the dataset original_thic_resample and the real MNIST.

<p align="center">
    <b>Augmented MNIST &emsp; &emsp; &emsp; | &emsp; &emsp; &emsp; Original MNIST</b>
</p>
<p align="center">
    <img src="https://github.com/RomainGratier/Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/blob/master/references/original_thic_resample.png" alt="Snow" style="width:100%" width="300" height="300" title="Augmented MNIST" >
    <img src="https://github.com/RomainGratier/Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/blob/master/references/original.png" alt="Forest" style="width:100%" width="300" height="300" title="Original MNIST">
</p>

## Experiences
We created two similar experiences to assess our model quality. Using our augmented MNIST dataset we can create a model that will try to optimize the stroke width of a digit. During the first experiment, the stroke width of a digit is increased compared to our previous knowledge about our data. Hence, the UMIN will be trained on digit with strokes width between 0 and 6 pixels [pxl] and it will try to create digits with higher stroke width. During the second experiment, the stroke width of the digits will be minimized. Hence, UMIN will be trained on digit with strokes width between 2.5 and 8 pixels [pxl] and it will try to create digits with lower stroke width.

## Generator training
The following figure displays samples from the true distribution from an augmented MNIST dataset.

**Real samples:**

<p align="center">
    <b> &emsp; &emsp; &ensp; Maximization experiment &emsp; &emsp; &emsp; | &emsp; &emsp; &emsp; Real samples &emsp; &emsp; &emsp; |  &emsp; &emsp; &emsp; Minimization experiment &emsp; &emsp; &emsp; &emsp;</b>
</p>
<p align="center">
  <img src="https://github.com/RomainGratier/Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/blob/master/references/gan_training_iterations_full_wgan_max.gif" width="280" height="280" alt="Sublime's custom image" title="Maximization"/>
  <img src="https://github.com/RomainGratier/Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/blob/master/references/real_mnist.png?raw=true" width="250" height="250" alt="Sublime's custom image" title="Real samples"/>
  <img src="https://github.com/RomainGratier/Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/blob/master/references/gan_training_iterations_full_wgan_min.gif" width="280" height="280" alt="Sublime's custom image" title="Minimization"/>
</p>

## Quantitative Results
Here are displayed our results in terms of prediction precision with the mean squared error and mean relative error.

<p align="center">
    <b>MSE/MRE</b>
</p>
<p align="center">
  <img src="https://github.com/RomainGratier/Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/blob/master/references/mse.png?raw=true" alt="Sublime's custom image" title="Real samples"/>
  <em>Real samples</em>
</p>

The next table exhibit the quantitative assessment of the data quality using the frêchet inception distance (FID) and the kernel inception distance (KID).

<p align="center">
    <b>FID/KID</b>
</p>
<p align="center">
  <img src="https://github.com/RomainGratier/Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/blob/master/references/fid.png?raw=true" alt="Sublime's custom image" title="Real samples"/>
</p>

## Qualitative Results
To assess our results. It is interesting to compare the best sample selected by our inference function and the worst samples discarded by our inference function. The first two rows display the generated samples and the last two rows exhibit real samples with the targeted width stroke.

<p align="center">
    <b>Best samples</b>
</p>
<p align="center">
  <img src="https://github.com/RomainGratier/Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/blob/master/references/wgan_min_qualitative_bayesian.png?raw=true" alt="Sublime's custom image" title="Real samples"/>
</p>

<p align="center">
    <b>Worst samples</b>
</p>
<p align="center">
  <img src="https://github.com/RomainGratier/Black-box_Optimization_via_Deep_Generative-Exploratory_Networks/blob/master/references/wgan_min_qualitative_bayesian_worse.png?raw=true" alt="Sublime's custom image" title="Real samples"/>
</p>

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us. 

## Authors

* **Romain Gratier de Saint-Louis** [PurpleBooth](https://github.com/RomainGratier)
* **Yuejiang Liu** - *VITA Lab EPFL* - [PurpleBooth](https://github.com/YuejiangLIU)
* **Alexandre Alahi** - *VITA Lab EPFL*

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Special thanks to the writers of the paper available on arxiv: https://arxiv.org/pdf/1912.13464.pdf. Their contribution motivates this project.

Another special thanks to writers of the paper available on arxiv ( https://arxiv.org/pdf/1901.02731.pdf ) and their github repository to allow us to use accurate Bayesian neural networks.

Bravo to the work done on https://github.com/dccastro/Morpho-MNIST that allows us to find an interesting dataset and established ground-truth measurements for our generated data

Bravo also to https://github.com/eriklindernoren/PyTorch-GAN to share the PyTorch implementation of useful GANs

And Bravo to https://github.com/abdulfatir/gan-metrics-pytorch.git that implemented a great FID score for MNIST dataset
