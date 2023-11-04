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

import shapely
from shapely.geometry import LineString, Point
import copy

def div_cast(x, m=100):
    for i in range(len(x)):
        e = x[i]
        e = round(e * m)
        e = e / m
        x[i] = e
    return x

def obs_check(obstacle, otype, before_pos, agent_pos):
    agent_ray = LineString([(before_pos[0], before_pos[1]), (agent_pos[0], agent_pos[1])])

    intersect = obstacle.intersection(agent_ray)

    if not 'EMPTY' in str(intersect) and 'LINESTRING' in str(intersect):

        print('intersect', intersect)
        print('obstacle', obstacle)
        print('before-pos', before_pos)
        print('agent-pos', agent_pos)
        raise Exception('done')

    if not 'EMPTY' in str(intersect) and not 'LINESTRING' in str(intersect):
        if otype == 'vertical':
            if before_pos[0] <= agent_pos[0]:
                delta = -0.01
            else:
                delta = 0.01
            agent_pos[0] = div_cast([intersect.x])[0] + delta

        elif otype == 'horizontal':
            if before_pos[1] <= agent_pos[1]:
                delta = -0.01
            else:
                delta = 0.01
            agent_pos[1] = div_cast([intersect.y])[0] + delta
        else:
            raise Exception()

        #agent_pos[0] = intersect.x + delta
        #agent_pos[1] = intersect.y + delta

    return agent_pos[:]
    
def obstacle_detection(obs_list, otype_list, before_pos, agent_pos):
    # find intersections of obstacles and the movement ray, and pick the closest intersection to the starting point
    n = len(obs_list)
    agent_ray = LineString([(before_pos[0], before_pos[1]), (agent_pos[0], agent_pos[1])])

    intersections = []
    intersection_type = []
    for i in range(n):
        obstacle = obs_list[i]
        intersect = obstacle.intersection(agent_ray)

        if not 'EMPTY' in str(intersect) and 'LINESTRING' in str(intersect):
            print('intersect', intersect)
            print('obstacle', obstacle)
            print('before-pos', before_pos)
            print('agent-pos', agent_pos)
            raise Exception('done')

        if not 'EMPTY' in str(intersect) and not 'LINESTRING' in str(intersect):
            intersections.append(np.array([intersect.x, intersect.y]).astype('float32'))
            intersection_type.append(otype_list[i])

    if len(intersections) == 0:
        return agent_pos 
    else:
        diff = [np.linalg.norm(item - before_pos, 2) for item in intersections]
        inter_sec_idx = np.argmin(diff)
        inter_sec = intersections[inter_sec_idx]
        otype = intersection_type[inter_sec_idx]
        
        update_pos = div_cast(inter_sec)
        if otype == 'vertical':
            if before_pos[0] <= agent_pos[0]:
                delta = -0.01
            else:
                delta = 0.01
            update_pos[0] = update_pos[0] + delta

        elif otype == 'horizontal':
            if before_pos[1] <= agent_pos[1]:
                delta = -0.01
            else:
                delta = 0.01
            update_pos[1] = update_pos[1] + delta
        else:
            raise Exception() 

        return update_pos[:]

class RoomObstacleEnv:

    def __init__(self):
        self.agent_pos = [0, 0]

    def random_action(self):

        delta = [0, 0]
        delta[0] = random.uniform(-0.2, 0.2)
        delta[1] = random.uniform(-0.2, 0.2)

        return div_cast(delta)

    def step(self, a):

        self.agent_pos = self.agent_pos[:]

        before_pos = copy.deepcopy(self.agent_pos[:])

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

        obs_lst = []
        #obs_lst.append(LineString([(0.63,0.25), (0.63, 0.75)]))

        delta = 0.01
        obs_lst.append(LineString([(0.501,0.001 - delta), (0.501, 0.401)]))
        obs_lst.append(LineString([(0.501,0.601 - delta), (0.501, 1.01)]))
        obs_lst.append(LineString([(0.201,0.401), (0.801, 0.401)]))
        obs_lst.append(LineString([(0.201,0.601), (0.801, 0.601)]))

        otype_lst = ['vertical', 'vertical', 'horizontal', 'horizontal']
        #agent_ray = LineString([(before_pos[0], before_pos[1]), (self.agent_pos[0], self.agent_pos[1])])
        
        #intersect = obstacle.intersection(agent_ray)

        #if not 'EMPTY' in str(intersect) and not 'LINESTRING' in str(intersect):
        #    if before_pos[0] <= self.agent_pos[0]:
        #        delta = -0.01
        #    else:
        #        delta = 0.01

        #    self.agent_pos[0] = intersect.x + delta
        #    self.agent_pos[1] = intersect.y + delta
        agent_pos = copy.deepcopy(self.agent_pos)
        self.agent_pos = obstacle_detection(obs_lst, otype_lst, before_pos, agent_pos)

        # for j in range(len(obs_lst)):
        #     self.agent_pos = obs_check(obs_lst[j], otype_lst[j], before_pos, self.agent_pos)

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

        agent_pos = copy.deepcopy(self.agent_pos)
        x[min(99, int(round(agent_pos[0] * 100))), min(99, int(round(agent_pos[1] * 100)))] += 1

        # x = self.blur_obs(x)

        # x = np.concatenate([x, self.exo.reshape((self.m,self.m))], axis=1)

        exo = [0.0, 0.0]

        return x.flatten(), agent_pos, exo


if __name__ == "__main__":

    env = RoomEnv()

    for i in range(0, 5000):
        a = env.random_action()
        print('s', env.agent_pos)
        print('a', a)
        x, _, _ = env.get_obs()
        print('x-argmax', x.argmax())
        env.step(a)
