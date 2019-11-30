import gym 
env = gym.make('CartPole-v0')
env.reset() 
for _ in range(1000):  
   env.render()   
   observation_n, reward_n, done_n, info = env.step(env.action_space.sample())
   print("info", observation_n, reward_n, done_n, info )

   # In the case of CartPole reward == 1 while still alive.  


   #if done_n: #Resets pole
   #    env.reset()
