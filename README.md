# 4106-RL

After an absurd amount of time, here's how to set up so we can easily work together.


## Set up
1. You need Anaconda (and not docker, apparently)

To install Anaconda (Windows), first uninstall all of python. Also, remove all python user/system variables.

Next, download [Anaconda](https://www.anaconda.com/distribution/#download-section) 3.7, 64-bit. Install for all users (otherwise, you'll have issues with doing jupyter notebook stuff, external to this project), and let it add the variables.


2. Clone this repo, then navigate to it in cmd

execute the following. It will probably take a while. 

```
conda env create --file environment.yml
```

Now, when you do the following

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

Currently, the only way to end the program is to shut down the executing program and close the display separately. 


## To exit

```
conda deactivate
```


