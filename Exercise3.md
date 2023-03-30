# Part 1

# Training Convolutional Nets

In this exercise we will look more into ConvNets

As discussed in the lecture, the key ideas to convnets are local connectivity and weight sharing which lead us to convolutions.

Before trying to *learn* convolutional filters, let's manually construct them

## Step 1 - Image Processing

Write a function that downloads an image at a given URL and converts it into greyscale and returns a 2D array.

Tip: use the `requests` and the `PIL` python libraries

## Step 2 - the `torch.nn.Conv2D` object

Research how to access the parameters of the convolution kernel module `torch.nn.Conv2D` in PyTorch and make sure you understand where 
kernel size, input channels and output channels come into definint the shape of the parameter tensors.


## Step 3 - Manually creating Convolutions

Create a `Conv2D` module with 1 input channel and 2 output channels with kernel size 3

* set the weights of of first output channel is a vertical edge detector
* set the weights of the second output channel is a horizontal edge detector
* create a size 1 input batch with a greyscale image of your choice and visualize the outputss

## Step 4 - Use a pre-trained network do analyze an image of your choice

* Check out the following link, and try using the famour `inception_v3` model to process an image of your choice
* https://pytorch.org/hub/pytorch_vision_inception_v3/


# Part 2

## Step 1 - Download the MNIST Dataset

The "Hello World" of convolutional neural networks are MNIST images. You can get the training data by
running the following command in `pip`

```
pip install mnist
```

and get e.g. the training data (there is a separate validation dataset). 

Here is simple code to sample minibatches from the training of validation dataset (not the simple preprocessing)

```
import mnist
train_labels = mnist.train_labels()
test_labels = mnist.test_labels()
scale_mean, scale_std = mnist.train_images().mean(),mnist.train_images().std()
train_images = (mnist.train_images()-scale_mean)/scale_std
test_images = (mnist.test_images()-scale_mean)/scale_std

def make_batch(N = 300, collection = 'train'):
    images = train_images if collection == 'train' else test_images
    labels = train_labels if collection == 'train' else test_labels
        
    indices = np.random.choice(np.arange(len(images)), size = (N,), replace = True)
    X = images[indices]
    y = labels[indices]
    return torch.FloatTensor(X[:,None]),torch.LongTensor(y)
```

## Step 2 - Make a CNN

Create a ConvNet with the following structure

* Conv 5 x 5 ( 1 -> 16 channels) -> ReLU -> MaxPool 2 x 2
* Conv 3 x 3 ( 16 -> 16 channels) -> ReLU -> MaxPool 2 x 2
* Conv 2 x 2 ( 16 -> 32 channels) 


Find out what the output on a random MNIST-like torch tensor is, i.e. `x = torch.randn(123,1,28,28)`

* Use `torch.nn.Flatten` to flatten all the remaining dimensions after the three convolutions into

How big is this intermediate representation of an image?

Finish the network by adding a head that ends with a `Linear(N,10)` layer (which we interpret to be the logits of the per-class probabilities).

With such an output you can then evaluate the multi-class loss via (the softmax is done internally) 

```
p = model(X)
loss = torch.nn.functional.cross_entropy(p,y)
````

## Step 3 - Train a CNN on MNIST

Use this model now to train your neural network to recognize the digit in these 28,28 images

* Check out the following link, and try using the famour `inception_v3` model to process an image of your choice
* https://pytorch.org/hub/pytorch_vision_inception_v3/
