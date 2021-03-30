#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import gym
import random
import time
from IPython.display import clear_output


# In[25]:


env=gym.make("FrozenLake-v0")


# In[26]:


action_space_size=env.action_space.n
state_space_size=env.observation_space.n

q_table=np.zeros((state_space_size,action_space_size))
print(q_table)


# In[34]:


num_episodes=10000
max_steps_per_episode=100

learning_rate=0.1
discount_rate=0.99

exploration_rate=1
max_exploration_rate=1
min_exploration_rate=0.01
exploration_decay_rate=0.002


# In[35]:


rewards_all_episodes=[]

#Q-learning algo
for episode in range(num_episodes):
    state=env.reset()
    rewards_current_episode=0
    done=False
    for step in range(max_steps_per_episode):
        exploration_rate_threshold=random.uniform(0,1)
        
        if(exploration_rate_threshold>exploration_rate):
            action=np.argmax(q_table[state,:])
        else:
            action=env.action_space.sample()
        
        new_state,reward,done,info=env.step(action)
        
        q_table[state,action]=q_table[state,action]*(1-learning_rate)+                            learning_rate*(reward+discount_rate*np.max(q_table[new_state,:]))
        
        state=new_state
        rewards_current_episode+=reward
        
        if done==True:
            break
    
    exploration_rate=min_exploration_rate+(max_exploration_rate-min_exploration_rate)*                    np.exp(-exploration_decay_rate*episode)
    
    rewards_all_episodes.append(rewards_current_episode)
    
#calculate and print the avg reward per 1000 eps
rewards_per_thousand_episodes=np.split(np.array(rewards_all_episodes),num_episodes/1000)
count = 1000
print("**********Average reward per 1000 eps************\n")
for r in rewards_per_thousand_episodes:
    print(count,": ",str(sum(r/1000)))
    count+=1000


# In[42]:


#testing agent 
for ep in range(3):
    state=env.reset()
    print("****Episode :",ep+1,"****\n\n\n")
    time.sleep(1)
    
    for step in range(max_steps_per_episode):
        clear_output(wait=True)
        env.render()
        time.sleep(0.3)
        action=np.argmax(q_table[state,:])
        state,reward,done,info=env.step(action)
        if done:
            clear_output(wait=True)
            env.render()
            if reward==1:
                print("\n****You Reached The Goal🥳****")
            else:
                print("\n****You Fell In A Hole😵****")
            time.sleep(3)
            clear_output(wait=True)
            break
env.close()


# In[ ]:




