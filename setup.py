from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Deep neural networks have garnered tremendous excitement in recent years thanks to their superior learning capacity in the presence of abundant data resources. However, collecting an exhaustive dataset covering all possible scenarios is often slow, expensive, and even impractical. The goal of this project is to devise a new learning framework that can learn from a finite dataset and noisy feedback of data properties to discover novel samples of particular interest. We will design and implement algorithms to interweave emerging deep generative modelling with classical Markov decision processes. We will evaluate our method in comparison to existing approaches through extensive experiments, including but not limited to visual semantic extrapolation and natural adversarial examples in the context of autonomous vehicles.',
    author='romain gratier',
    license='MIT',
)
