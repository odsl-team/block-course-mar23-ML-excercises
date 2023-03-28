# Distinguising Supersymmetric Particles from Background

Let's do some physics! Instead of classfying cats and dogs, we will use deep neural networks to distinguish cthat ollisions from the Large Hadron Collider contain supersymmetric particles from events that emerge purely from standard model physics.

We will use a small version of the the [UCI SUSY Dataset](https://archive.ics.uci.edu/ml/datasets/SUSY), which we will use. The repository has small csv file with 30k events.

This dataset was one of the early ones that used deep learning in High Energy Physics and originates from this paper

Baldi, P., P. Sadowski, and D. Whiteson. “Searching for Exotic Particles in High-energy Physics with Deep Learning.” Nature Communications 5 (July 2, 2014)


https://www.nature.com/articles/ncomms5308

## Step 1: Load and Split the Data

1. Write a function `load_and_split_data(filename, train_ratio)` whose functionality is to load and split the data for a specific ratio of train and validation data (we won't use a third test dataset for now, which we would need to make a final assessment). Use this function to split your data with 70% training data and 30% validation data

2. Write a function `make_data(dataset, N)` that samples randomly (with replacement) from a dataset (either the training or the validation dataset)

## Step 2: Visualize the Dataset

It's always good to visualize the data before starting to do crazy machine learning with it. This is
sometimes referred to as exploratory data analysis (EDA) and is useful to catch e.g. vastly diverging
scales between features

Visualize the 18 data features on 18 panels, where on each panel you have two histograms
* the distribution of the feature for y == 0
* the distribution of the feature for y == 1

## Step 3: Create a function that creates a simple MLP

The neural network should have width N hidden layers and use ReLU activations internally and finish up with a sigmoid activation. Use the standard `torch` building blocks and `torch.nn.Sequential` to build up the model.

A suggestions: wrap the creation of the neural network into a function `make_model(N)` to be able to easily modify the number of units inside of the hidden layers of the network

## Step 4: Create a Training Loop

Now let's train the neural network! Write a function `train(dataset)` that does the following steps

1. create the neural network
2. create an optimizer like Adam (with learning rate like 1e-3
3. Loop over N iterative steps (e.g. N=3000)

For each step of the training loop

1. Sample a minibatch of data and labels
2. Evaluate the neural network to get predictions
3. use the prediction and the true labels to evaluate the loss
4. trigger backpropagation

## Step 5: Train!

Use the training loop function and plot the evolution of the loss as a function of the training steps

## Step 6: Wait a minute - Evaluate the True Performance

The last training loss at the end of the training procedure will not be an accurate reflection of the true model performance.

Write a function `eval_model(model, data)` that evaluates the performance of the model without tracking gradients (you can use the `with torch.no_grad()` functionality)

Use this function to derive an estimate of the neural network on the validation data

## Step 7: 👀 - seems like we can't trust the train performance

You will likely have found that the performance on the validation data is much worse than the performance on the training data. You were overfitting without really knowing it. Let's adjust the training loop such that it always tracks both the training loss as well as the validation loss.

Plot the training and validation loss as a function of the training steps.

At which point would you stop training? (You can also add logic to save the best validation loss model using `torch.save`)

## Step 8: Getting the ROC curve

Given that we have a trained model we can now look at its performance

Write a function `plot_distribution` that creates a 2-pane plot

1. One one pane plot the output of the neural network as histograms for 
    * data originating from standard model events
    * data originating from supersymmetric events
    
2. Install scikit-learn via `pip install sklearn` and compute the false positive and true positive
   rates via the `sklearn.roc_curve` function and plot the ROC curve

6. do the optimization step
7. IMPORTANT: use model.zero_grad() to flush the gradient information

Tip: It might be nice to track the trajectory of loss values as you are training. You can add simple logic to this
