# 4106-RL

After an absurd amount of time, here's how to set up so we can easily work together.

This environment should be able to handle developing RL algorithms to run on any of the [openai default environments](https://github.com/openai/gym/wiki/Table-of-environments?fbclid=IwAR013H67TnteguyIg3gW5ASQ_RxoBCowMWVBSIXec0-pymVGnXF-msVLbp4) on Windows.

Currently, none of the atari ones work. 

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




## Other

Download http://www.msys2.org/ (x86-64)

within its terminal, run
```
pacman -Su
```

then run
```
pacman -S base-devel mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake
```
with all. 

## References
https://github.com/j8lp/atari-py
http://www.msys2.org/
http://ronny.rest/tutorials/module/openai_001/openai_first_test/?fbclid=IwAR3ppq-jZXdK82_wUcWLCiZdvP4vepaIcEcvTf9rSjBV2rp8CNvDdfD9Q-g
(Book, uottawa online resource) Hands-On Reinforcement Learning with Python: Master reinforcement and deep reinforcement learning using OpenAI Gym and TensorFlow By Sudharsan Ravichandiran