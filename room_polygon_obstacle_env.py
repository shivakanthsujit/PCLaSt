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
from shapely.geometry import LineString, Point, Polygon
import copy

def div_cast(x, m=100):
    for i in range(len(x)):
        e = x[i]
        e = round(e * m)
        e = e / m
        x[i] = e
    return x



def add_polygon_walls(line_ends, obs_type, delta = 0.00):
    # line_ends = [(x_1, y_1), (x_2, y_2)] where (x, y) are the coordinates of the end points of a line
    # order requirement: x_1 <= x_2, y_1 <= y_2
    # x and y need to have exactly two digits, e.g., (x, y) = (0.10, 0.15)
    # obs_type = 'vertical' or 'horizontal'
    # delta denotes the width of the wall and has to be an integer times 0.1
    if obs_type == 'vertical':
        x_1, y_1 = line_ends[0]
        x_2, y_2 = line_ends[1]
        delta_modified = delta + 0.009
        obs = Polygon([(x_1 - delta_modified, y_1 - delta_modified), (x_2 - delta_modified, y_2 + delta_modified),
                       (x_2 + delta_modified, y_2 + delta_modified), (x_1 + delta_modified, y_1 - delta_modified)])

    elif obs_type == 'horizontal':
        x_1, y_1 = line_ends[0]
        x_2, y_2 = line_ends[1]
        delta_modified = delta + 0.009
        obs = Polygon([(x_1 - delta_modified, y_1 - delta_modified), (x_1 - delta_modified, y_1 + delta_modified),
                       (x_2 + delta_modified, y_2 + delta_modified), (x_2 + delta_modified, y_2 - delta_modified)])

    else:
        raise NotImplementedError
    
    return obs, obs_type

def obstacle_detection(obs_list, otype_list, before_pos, agent_pos):
    # find intersections of obstacles and the movement ray, and pick the closest intersection to the starting point
    n = len(obs_list)

    agent_ray = LineString([(before_pos[0], before_pos[1]), (agent_pos[0], agent_pos[1])])

    intersections = []
    intersection_type = []
    for i in range(n):
        obstacle = obs_list[i]
        intersect = obstacle.intersection(agent_ray)

        if not 'EMPTY' in str(intersect):
            intersect_points_list = [np.array(item).astype('float32') for item in list(intersect.coords)]
            intersect_type_list = [otype_list[i]]*len(intersect_points_list)

            intersections = intersections + intersect_points_list
            intersection_type = intersection_type + intersect_type_list

    if len(intersections) == 0:
        return agent_pos[:]
    else:
        diff = [np.linalg.norm(item - before_pos, 2) for item in intersections]
        inter_sec_idx = np.argmin(diff)
        inter_sec = intersections[inter_sec_idx]
        otype = intersection_type[inter_sec_idx]
        
        update_pos = div_cast(inter_sec)

        # if otype == 'vertical':
        #     if before_pos[0] <= agent_pos[0]:
        #         delta = -0.01
        #     else:
        #         delta = 0.01
        #     update_pos[0] = update_pos[0] + delta

        # elif otype == 'horizontal':
        #     if before_pos[1] <= agent_pos[1]:
        #         delta = -0.01
        #     else:
        #         delta = 0.01
        #     update_pos[1] = update_pos[1] + delta
        # else:
        #     raise Exception() 
        
        return update_pos[:]

class RoomPolygonObstacleEnv:

    # def __init__(self):
    #     self.agent_pos = [0, 0]

    #     obs_lst = []

    #     obs_lst.append(Polygon([(0.491, -0.04), (0.491, 0.399),
    #                             (0.509, 0.399), (0.509, -0.04)]))
    #     obs_lst.append(Polygon([(0.491, 0.601), (0.491, 1.005),
    #                             (0.509, 1.005), (0.509, 0.601)]))
    #     obs_lst.append(Polygon([(0.201, 0.391), (0.201, 0.409),
    #                             (0.799, 0.409), (0.799, 0.391)]))
    #     obs_lst.append(Polygon([(0.201, 0.591), (0.201, 0.609),
    #                             (0.799, 0.609), (0.799, 0.591)]))
        
    #     otype_lst = ['vertical', 'vertical', 'horizontal', 'horizontal']
        
    #     self.obs_lst = obs_lst
    #     self.otype_lst = otype_lst


    def __init__(self):
        self.agent_pos = [0, 0]

        wall_lst = [ [(0.50, 0.00), (0.50, 0.40)],
                    [(0.50, 0.60), (0.50, 1.00)],
                    [(0.20, 0.40), (0.80, 0.40)],
                    [(0.20, 0.60), (0.80, 0.60)]
                    ]
        otype_lst = ['vertical', 'vertical','horizontal', 'horizontal']

        obs_lst = []
        for idx, end_points in enumerate(wall_lst):
            obs, _ = add_polygon_walls(end_points, otype_lst[idx])
            obs_lst.append(obs)

        self.obs_lst = obs_lst
        self.otype_lst = otype_lst


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

        obs_lst, otype_lst = self.obs_lst, self.otype_lst
        
        agent_pos = copy.deepcopy(self.agent_pos[:])
        updated_pos = obstacle_detection(obs_lst, otype_lst, before_pos, agent_pos)

        self.agent_pos = div_cast(updated_pos)[:]


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

class RoomMultiPassageEnv:

    def __init__(self):
        self.agent_pos = [0, 0]

        wall_lst = [ [(0.25, 0.00), (0.25, 0.80)],
                    [(0.25, 0.90), (0.25, 1.00)],
                    [(0.50, 0.00), (0.50, 0.20)],
                    [(0.50, 0.30), (0.50, 1.00)],
                    [(0.75, 0.00), (0.75, 0.45)],
                    [(0.75, 0.55), (0.75, 1.00)] ]
        otype_lst = ['vertical']*6

        obs_lst = []
        for idx, end_points in enumerate(wall_lst):
            obs, _ = add_polygon_walls(end_points, otype_lst[idx])
            obs_lst.append(obs)

        self.obs_lst = obs_lst
        self.otype_lst = otype_lst

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

        obs_lst, otype_lst = self.obs_lst, self.otype_lst

        agent_pos = copy.deepcopy(self.agent_pos[:])
        updated_pos = obstacle_detection(obs_lst, otype_lst, before_pos, agent_pos)

        self.agent_pos = div_cast(updated_pos)[:]

    def blur_obs(self, x):
        x = torch.Tensor(x).unsqueeze(0).unsqueeze(0)
        x = F.gaussian_blur(x, 7) * 16
        x = x.squeeze(0).squeeze(0).numpy()
        return x

    def synth_obs(self, ap):
        x = np.zeros(shape=(1, 100, 100))
        x[:, int(round(ap[0] * 100)), int(round(ap[1] * 100))] += 1

        # x = self.blur_obs(x)

        return x

    def get_obs(self):
        x = np.zeros(shape=(1, 100, 100))

        agent_pos = copy.deepcopy(self.agent_pos)
        x[:, min(99, int(round(agent_pos[0] * 100))), min(99, int(round(agent_pos[1] * 100)))] += 1

        # x = self.blur_obs(x)

        # x = np.concatenate([x, self.exo.reshape((self.m,self.m))], axis=1)

        exo = [0.0, 0.0]

        return x, agent_pos, exo


class RoomSpiral:

    def __init__(self):
        self.agent_pos = [0, 0]

        wall_lst = [ [(0.20, 0.00), (0.20, 0.75)],
                    [(0.20, 0.75), (0.80, 0.75)],
                    [(0.80, 0.25), (0.80, 0.75)],
                    [(0.40, 0.25), (0.80, 0.25)],
                    [(0.40, 0.25), (0.40, 0.50)],
                    [(0.40, 0.50), (0.60, 0.50)] ]
        otype_lst = ['vertical', 'horizontal', 'vertical', 'horizontal', 'vertical', 'horizontal']

        obs_lst = []
        for idx, end_points in enumerate(wall_lst):
            obs, _ = add_polygon_walls(end_points, otype_lst[idx])
            obs_lst.append(obs)

        self.obs_lst = obs_lst
        self.otype_lst = otype_lst

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

        obs_lst, otype_lst = self.obs_lst, self.otype_lst

        agent_pos = copy.deepcopy(self.agent_pos[:])
        updated_pos = obstacle_detection(obs_lst, otype_lst, before_pos, agent_pos)

        self.agent_pos = div_cast(updated_pos)[:]

    def blur_obs(self, x):
        x = torch.Tensor(x).unsqueeze(0).unsqueeze(0)
        x = F.gaussian_blur(x, 7) * 16
        x = x.squeeze(0).squeeze(0).numpy()
        return x

    def synth_obs(self, ap):
        x = np.zeros(shape=(1, 100, 100))
        x[:, int(round(ap[0] * 100)), int(round(ap[1] * 100))] += 1

        # x = self.blur_obs(x)

        return x

    def get_obs(self):
        x = np.zeros(shape=(1, 100, 100))

        agent_pos = copy.deepcopy(self.agent_pos)
        x[:, min(99, int(round(agent_pos[0] * 100))), min(99, int(round(agent_pos[1] * 100)))] += 1

        # x = self.blur_obs(x)

        # x = np.concatenate([x, self.exo.reshape((self.m,self.m))], axis=1)

        exo = [0.0, 0.0]

        return x, agent_pos, exo
    

if __name__ == "__main__":

    env = RoomPolygonObstacleEnv()

    for i in range(0, 5000):
        a = env.random_action()
        print('s', env.agent_pos)
        print('a', a)
        x, _, _ = env.get_obs()
        print('x-argmax', x.argmax())
        env.step(a)


class RoomMultiPassageEnvLarge:

    def __init__(self):
        self.agent_pos = [0, 0]

        wall_lst = [ [(0.25, 0.00), (0.25, 0.70)],
                    [(0.25, 0.90), (0.25, 1.00)],
                    [(0.50, 0.00), (0.50, 0.20)],
                    [(0.50, 0.40), (0.50, 1.00)],
                    [(0.75, 0.00), (0.75, 0.35)],
                    [(0.75, 0.55), (0.75, 1.00)] ]
        otype_lst = ['vertical']*6

        obs_lst = []
        for idx, end_points in enumerate(wall_lst):
            obs, _ = add_polygon_walls(end_points, otype_lst[idx])
            obs_lst.append(obs)

        self.obs_lst = obs_lst
        self.otype_lst = otype_lst

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

        obs_lst, otype_lst = self.obs_lst, self.otype_lst

        agent_pos = copy.deepcopy(self.agent_pos[:])
        updated_pos = obstacle_detection(obs_lst, otype_lst, before_pos, agent_pos)

        self.agent_pos = div_cast(updated_pos)[:]

    def blur_obs(self, x):
        x = torch.Tensor(x).unsqueeze(0).unsqueeze(0)
        x = F.gaussian_blur(x, 7) * 16
        x = x.squeeze(0).squeeze(0).numpy()
        return x

    def synth_obs(self, ap):
        x = np.zeros(shape=(1, 100, 100))
        x[:, int(round(ap[0] * 100)), int(round(ap[1] * 100))] += 1

        # x = self.blur_obs(x)

        return x

    def get_obs(self):
        x = np.zeros(shape=(1, 100, 100))

        agent_pos = copy.deepcopy(self.agent_pos)
        x[:, min(99, int(round(agent_pos[0] * 100))), min(99, int(round(agent_pos[1] * 100)))] += 1

        # x = self.blur_obs(x)

        # x = np.concatenate([x, self.exo.reshape((self.m,self.m))], axis=1)

        exo = [0.0, 0.0]

        return x, agent_pos, exo

