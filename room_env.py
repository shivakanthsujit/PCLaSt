'''
    Environment where agent moves in the (0,255), where the state is kept to always be divisible by 256.  

    agent_pos: 2-vector

    x,y

    render function to image.  Place a pixel in a 256x256 image where the agent is.  

    a - left (0), right (1), up (2), down (3), same (4)

'''

import numpy as np
import random
import torchvision.transforms.functional as F
import torch


def div_cast(x, m=100):
    for i in range(len(x)):
        e = x[i]
        e = round(e * m)
        e = e / m
        x[i] = e
    return x


class RoomEnv:

    def __init__(self):
        self.agent_pos = [0, 0]

    def random_action(self):

        delta = [0, 0]
        delta[0] = random.uniform(-0.2, 0.2)
        delta[1] = random.uniform(-0.2, 0.2)

        return div_cast(delta)

    def step(self, a):

        self.agent_pos = self.agent_pos[:]

        self.agent_pos = div_cast(self.agent_pos)

        self.agent_pos[0] += a[0]
        self.agent_pos[1] += a[1]

        if self.agent_pos[0] <= 0:
            self.agent_pos[0] = 0
        if self.agent_pos[0] >= 1:
            self.agent_pos[0] = 0.99

        if self.agent_pos[1] <= 0:
            self.agent_pos[1] = 0
        if self.agent_pos[1] >= 1:
            self.agent_pos[1] = 0.99

        self.agent_pos = div_cast(self.agent_pos)[:]

    def blur_obs(self, x):
        x = torch.Tensor(x).unsqueeze(0).unsqueeze(0)
        x = F.gaussian_blur(x, 7) * 16
        x = x.squeeze(0).squeeze(0).numpy()
        return x

    def synth_obs(self, ap):
        x = np.zeros(shape=(100, 100))

        x[int(round(ap[0] * 100)), int(round(ap[1] * 100))] += 1

        # x = self.blur_obs(x)

        return x.flatten()

    def get_obs(self):
        x = np.zeros(shape=(100, 100))

        x[int(round(self.agent_pos[0] * 100)), int(round(self.agent_pos[1] * 100))] += 1

        # x = self.blur_obs(x)

        # x = np.concatenate([x, self.exo.reshape((self.m,self.m))], axis=1)

        exo = [0.0, 0.0]

        return x.flatten(), self.agent_pos, exo


if __name__ == "__main__":

    env = RoomEnv()

    for i in range(0, 5000):
        a = env.random_action()
        print('s', env.agent_pos)
        print('a', a)
        x, _, _ = env.get_obs()
        print('x-argmax', x.argmax())
        env.step(a)
