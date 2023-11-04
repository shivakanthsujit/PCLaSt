
import gym
from gym import spaces
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
import cv2
import numpy as np
import torch

class RatGym(gym.Env):
    """
    Gym wrapper for RatInABox
    """

    def __init__(self):
        self.env = Environment()
        self.agent = Agent(self.env)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(100, 100, 3), dtype=np.float32
        )

    def render(self):
        grid_size = 200
        base = np.zeros((grid_size, grid_size, 3), dtype=np.uint8) + 200
        for wall in self.env.walls:
            wall = wall * grid_size
            cv2.line(
                base,
                (int(wall[0][0]), int(wall[0][1])),
                (int(wall[1][0]), int(wall[1][1])),
                (100, 100, 100),
                5,
            )
        agent_pos = self.agent.pos * grid_size
        cv2.circle(base, (int(agent_pos[0]), int(agent_pos[1])), 3, (50, 50, 200), -1)
        base = cv2.resize(base, (100, 100)) / 255.0
        return base

    def reset(self, start_pos=None):
        self.agent = Agent(self.env)
        if start_pos is not None:
            self.agent.pos = start_pos
        else:
            self.agent.pos = np.array([0.5, 0.5])
        return self.render().reshape(-1)

    def step(self, action):
        new_pos_goal = self.agent.pos + action
        #action_use = np.abs(new_pos_goal - self.agent.pos) * action

        for j in range(0,300):
            #action_use = np.abs(new_pos_goal - self.agent.pos) * action
            action_use = (new_pos_goal - self.agent.pos) * 100
            self.agent.update(drift_velocity=action_use)

        self.agent.velocity *= 0.0
        self.agent.velocity += 0.0001

        return (
            self.render().reshape(-1),
            0,
            False,
            {"pos": self.agent.pos, "vel": self.agent.velocity},
        )

    def add_wall(self, wall):
        self.env.add_wall(wall)


'''
Should match same interface as room_env
'''


def div_cast(x, m=100):
    for i in range(len(x)):
        e = x[i]
        e = round(e * m)
        e = e / m
        x[i] = e
    return x

import random
class RatEnvWrapper:

    def __init__(self): 

        self.env = RatGym()
        #self.env.add_wall([np.array([0.33, 0.25]), np.array([0.33, 0.75])])
        #self.env.add_wall([np.array([0.66, 0.25]), np.array([0.66, 0.75])])
        #self.env.add_wall([np.array([0.5, 0.25]), np.array([0.5, 0.75])])
        self.env.reset()

        init_action = np.array([0.0,0.0])
        self.step(init_action)

    def random_action(self):
        delta = [0, 0]
        delta[0] = random.uniform(-0.2, 0.2)
        delta[1] = random.uniform(-0.2, 0.2)
        return div_cast(delta)

    def step(self,action):
        obs, reward, done, info = self.env.step(action)
        self.last_obs = obs*1.0

        #print('-------------')
        #print('taken step with action', action)
        #print('agent pos', self.env.agent.pos)

    def get_obs(self):

        obs = self.last_obs*1.0
        
        obs = torch.Tensor(obs.reshape(1,100, 100, 3)).permute(0,3,1,2).reshape((1,3*100*100)).numpy()


        ast = div_cast(self.env.agent.pos)
        ast = torch.Tensor(ast).long().numpy()
        est = ast*0

        return obs, ast, est

    def synth_obs(self,ap):
        #print('self.env.agent.pos', type(self.env.agent.pos))
        #print('apply-pos', type(ap))
        #raise Exception('done')
        self.env.agent.pos = np.array(ap)
        init_action = np.array([0.0,0.0])
        self.step(init_action)
        obs,_,_ = self.get_obs()
        return obs

if __name__ == "__main__":

    from ratinabox.Environment import Environment
    from ratinabox.Agent import Agent
    import matplotlib.pyplot as plt
    import numpy as np
    from torchvision.utils import save_image
    import torch

    env = RatGym()
    #env.add_wall([np.array([0.33, 0.25]), np.array([0.33, 0.75])])
    #env.add_wall([np.array([0.66, 0.25]), np.array([0.66, 0.75])])
    env.add_wall([np.array([0.5, 0.25]), np.array([0.5, 0.75])])
    obs = env.reset()

    act = np.array([0.1, 0.1])
    print(env.agent.pos)
    #for j in range(0,10000):
    #    obs, reward, done, info = env.step(act)
    #    if j % 100 == 0:
    #        print(j)
    act = np.array([0.1, 0.0])
    obs, reward, done, info = env.step(act)
    obs = torch.Tensor(obs.reshape(1,100, 100, 3)).permute(0,3,1,2)

    print(env.agent.pos)

    save_image(obs, 'obs.png')


