import numpy as np
import csv
import random
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

def gen_vec(rand_range, size):
    if type(rand_range[0]) != list:
        rand_range = [rand_range]
    if len(rand_range) == 1:
        rand_range *= size
    elif len(rand_range) != size:
        raise ValueError("len(rand_range) != size")
    vec = []
    for v in rand_range:
        vec.append(random.uniform(*v))
    return vec
def get_min(func, input_values, stop_value):
    min_value = np.inf
    def no_stop():
        for v in input_values:
            min_value_temp = func(v)
            if min_value_temp < min_value:
                min_value = min_value_temp
        return min_value
    def stop():
        for v in input_values:
            min_value_temp = func(v)
            if min_value_temp == stop_value:
                return min_value_temp
            if min_value_temp < min_value:
                min_value = min_value_temp
        return min_value
    if stop_value == None:
        return no_stop()
    else:
        return stop()

# 2-link ROBOT
import math
def get_link_end_points(robot_link_len, rot1, rot2):
    x1 = robot_link_len[0]*math.cos(rot1)
    y1 = robot_link_len[0]*math.sin(rot1)
    x2 = x1 + robot_link_len[1]*math.cos(rot1+rot2)
    y2 = y1 + robot_link_len[1]*math.sin(rot1+rot2)
    return 0, 0, x1, y1, x2, y2

# =====================================================================================
import torch
from active import *
from diffco.Obstacles import Obstacle
# Set envs
DOF = 2
env_name = '1rect_active'
# Get Data (Robot & Obstacles)
dataset = torch.load('data/2d_{}dof_{}.pt'.format(DOF, env_name))
cfgs = dataset['data'].double() # 각도
labels = dataset['label'].reshape(-1, 1).double()
dists = dataset['dist'].reshape(-1, 1).double() 
obstacles = [list(o) for o in dataset['obs']]
obj_obstacles = [Obstacle(*param) for param in obstacles]
robot = dataset['robot'](*dataset['rparam'])
# Get FCL Obj of Obstacles
fcl_obs = [FCLObstacle(*param) for param in obstacles]
fcl_collision_obj = [fobs.cobj for fobs in fcl_obs]
# Get FCL Obj of Robot (Binary collision check)
obs_managers = [fcl.DynamicAABBTreeCollisionManager()]
obs_managers[0].registerObjects(fcl_collision_obj)
obs_managers[0].setup()
robot_links = robot.update_polygons(cfgs[0])
robot_manager = fcl.DynamicAABBTreeCollisionManager()
robot_manager.registerObjects(robot_links)
robot_manager.setup()
for mng in obs_managers:
    mng.setup()
# Obstacle Transform
position = torch.FloatTensor([-1.5000,  3.0000]) # New Obstacle Coordinate
obstacles[0][1] = (position[0].item(), position[1].item())
obj_obstacles = [Obstacle(*param) for param in obstacles]
fcl_collision_obj[0].setTransform(fcl.Transform([position[0], position[1], 0]))
for obs_mng in obs_managers:
    obs_mng.update()
# Get FCL checker
gt_checker = FCLChecker(obj_obstacles, robot, robot_manager, obs_managers)

# ======================================================================================
# Get Keyboard Input

from getkey import getkey, keys 
import numpy as np

t_speed = np.pi/20
def get_key_target(robot):
    global t_speed
    key = getkey()
    if key == keys.UP:
        print("UP")
        robot["config"][0] += t_speed
    elif key == keys.DOWN:
        print("DOWN")
        robot["config"][0] -= t_speed
    elif key == keys.RIGHT:
        print("RIGHT")
        robot["config"][1] += t_speed
    elif key == keys.LEFT:
        print("LEFT")
        robot["config"][1] -= t_speed
    print(robot["config"][1])
    time.sleep(0.1)
    key = None
    return robot

# ======================================================================================
# Draw Plot

import matplotlib.pyplot as plt
import time
from scipy.spatial.transform import Rotation

plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.set(xlabel='x', ylabel='y', title='Graphs')
ax1.grid()
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-7, 7]); ax1.set_ylim([-7, 7]); 

ax2 = fig.add_subplot(122)
ax2.grid()
ax2.set_aspect('equal', adjustable='box')
ax2.set_xlim([-np.pi, np.pi]); ax2.set_ylim([-np.pi, np.pi]); 

obj_x = [-2.5, -2.5, -0.5, -0.5, -2.5]
obj_y = [4, 2, 2, 4, 4]
line2, = ax1.plot(obj_x, obj_y)

robot = {"config": [0, 0]}
dot3, = ax2.plot([robot["config"][0]], [robot["config"][1]], 'bo')
robot_links = [3.5, 3.5]
x0, y0, x1, y1, x2, y2 = get_link_end_points(robot_links, *robot["config"])
x_data = [x0, x1, x2]
y_data = [y0, y1, y2]
line1, = ax1.plot(x_data, y_data)
_, dist =  gt_checker.predict(torch.Tensor([robot["config"]]), distance=True)
plt.legend([f"Config: {robot['config']}, Dist: {dist.item()}"], loc ="lower right")
fig.canvas.draw()
fig.canvas.flush_events()
while True:
    robot = get_key_target(robot)
    x0, y0, x1, y1, x2, y2 = get_link_end_points(robot_links, *robot["config"])
    x_data = [x0, x1, x2]
    y_data = [y0, y1, y2]
    line1.set_xdata(x_data); line1.set_ydata(y_data); 
    line2.set_xdata(obj_x); line2.set_ydata(obj_y); 
    dot3.set_xdata([robot["config"][0]]); dot3.set_ydata([robot["config"][1]]); 
    _, dist =  gt_checker.predict(torch.Tensor([robot["config"]]), distance=True)
    plt.legend([f"Config: {robot['config']}, Dist: {dist.item()}"], loc ="lower right")
    fig.canvas.draw()
    fig.canvas.flush_events()