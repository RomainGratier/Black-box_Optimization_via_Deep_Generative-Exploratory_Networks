# Get dataset morphoMNIST
git clone https://github.com/dccastro/Morpho-MNIST.git
cp -r Morpho-MNIST/morphomnist src/morphomnist
rm -rf Morpho-MNIST

# Get models for FID/KID measurement
git clone https://github.com/abdulfatir/gan-metrics-pytorch.git
cp -r gan-metrics-pytorch/models src/models
rm -rf gan-metrics-pytorch