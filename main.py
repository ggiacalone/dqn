from unityagents import UnityEnvironment
import numpy as np

import gym
import random
import torch

from collections import deque
import matplotlib.pyplot as plt
# %matplotlib inline


runner = Runner()


# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()




