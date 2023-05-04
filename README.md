# MarioKart

In this project we will implement an autonomous driving agent in SuperTuxKart (an open source version of MarioKart)

# Setup and Starter Code

The `MarioKartStarterColab.ipynb` file in this repo is the starter Colab notebook.

The `project` folder contains starter code for the project.

# Project Breakdown

This project consists of two main components: the controller and the planner.

## Controller

The controller is a library reponsible for implementing the actual driving logic (e.g. when to accelerate, how to turn, etc.) The controller takes in as input an "aim point" and the current velocity of the kart, and returns a `pystk` action (explained belwo) that dictates what the kart should do to move towards the aim point.

The aim point itself is an (x,y) point on the screen where (-1, -1) is the top left corner and (1,1) is the bottom right corner of the screen. The goal of the controller is to steer the kart towards the aim point. To do this, the controller returns a `pystk.Action`. 

`pystk` is the Python SuperTuxKart interface (see [repo](https://github.com/philkr/pystk)) that we will work with to develop our driving agent. A `pystk.Action`is just an object that the controller returns that should include the follwowing fields:
* `steer`: The steering angle of the kart - a float in the range [-1.0, 1.0]. 
  * A value of -1.0 means you want to turn the cart as far to the left as possible.
  * A value of 1.0 means you want to turn the cart as far to the right as possible.
* `acceleration`: The acceleration of the kart - a float in the range [0.0, 1.0]
  * A value of 0.0 means you do not want to accelerate at all.
  * A value of 1.0 means you want to accelerate as much as possible.
* `brake`: Whether or not to brake - a boolean.
* `drift`: Whether or not to drft - a boolean.
  * This can be useful in tight turns
* `nitro`: Whether or not to apply nitro for a boost of speed - a boolean.


Note: The controller does not incorporate any deep learning. Initially, the aim point provided to the controller is directly extracted from the SuperTuxKart simulator. Later, we will use a neural network to predict the aim point. We run the controller with the simulator to collect training data for the nerual network

Hints for implementing the controller:
* Add drift if the steering angle is too large (you’ll need to experiment to see what “too large” means).
* Choose a constant velocity you want to target
* Steering and relative aim use different scales. Use the aim point and a tuned scaling factor to choose the right amount of steering.

## Planner

The planner is where deep learning comes in. In this part you will train a Convolutional Neural Netowrk (CNN) to select an aim point on the image. This aim point will be passed to the controller to create a complete driving system.

### Data

We will use our controller to collect training data for your planner. WARNING: This creates a dataset in the folder drive_data. Make sure to back up any data that may be overwritten.

python -m homework.utils zengarden lighthouse hacienda snowtuxpeak cornfield_crossing scotland
You may experiment with the list of levels used to generate data, but we recommend this set. Adding additional levels may generate an unbalanced training set and lead to some issues with the test grader.

You can visualize your training data with

`python -m homework.visualize_data drive_data`

### Model

This is the part where we implement the Planner class in planner.py. The planner is a torch.nn.Module which takes as input an image tensor and outputs the aim point in the image coordinate (i.e., it should be a pair (x, y) of ints with x in [0, 127] and y in [0, 95]).

More instructions will be added here when we are ready to devlop and train the model.



