# 4106-RL

A repository for our CSI4106 final project: creating a reinforcement model to play a game, using varied NNs within it.

This project is designed to work in the given Anaconda environment, on Windows.  

## Creating/using the models

Currently, all of the work we've really done can be found in the directory algos.

You can run an entirely random game using

```
python reinforcement_learner.py
```

Otherwise, to create the individual models using the following

```
python reinforcement_learner.py {1|2|3} [maximum minutes] [maximum iterations] [maximum decreases in a row] [minimum number of epochs]
```
The [] are optional arguments, which indicate what they specify. Maximum minutes refers to the maximum number of minutes a single iteration of gathering data is allotted to gather said data. Each has a default value and so can be omitted as preferred.

As for the first argument, 
1 - Use Logistic regression
2 - Use MLP
3 - Use RNN


## Set up
1. If needed, install Anaconda

If you already have some versions of python installed, first uninstall them and remove all python user/system variables.

Next, download [Anaconda](https://www.anaconda.com/distribution/#download-section) 3.7, 64-bit. Install for all users (otherwise, you'll have issues with doing jupyter notebook stuff, external to this project), and let it add the variables.

2. Set up the environment
First, clone this repo, then navigate to it in a terminal.
Execute the following. It will probably take a while. 

```
conda env create --file environment.yml
```

Now, when you do the following command,

```
conda env list
```

One of the listed environments should be: `4106-RL` .  


3. Enter the environment

```
conda activate 4106-RL
```

There we go! 

And to make sure it's all good to go:

```
python test.py
```

This tests

## To exit

```
conda deactivate
```

