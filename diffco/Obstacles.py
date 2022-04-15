import torch
import fcl
import numpy as np

class Obstacle:
    def __init__(self, kind, position, size, cost=np.inf):
        self.kind = kind
        if self.kind not in ['circle', 'rect']:
            raise NotImplementedError('Obstacle kind {} not supported'.format(kind))
        self.position = torch.FloatTensor(position)
        self.size = torch.FloatTensor([size]) if kind == 'circle' else torch.FloatTensor(size)
        self.cost = cost
    
    def is_collision(self, point):
        if point.ndim == 1:
            point = point[np.newaxis, :]
        if self.kind == 'circle':
            return torch.norm(self.position-point, dim=1) < self.size/2
        elif self.kind == 'rect':
            return torch.all(torch.abs(self.position-point) < self.size/2, dim=1)
        else:
            raise NotImplementedError('Obstacle kind {} not supported'.format(self.kind))
    
    def get_cost(self):
        return self.cost

class FCLObstacle:
    def __init__(self, shape, position, size, rot, category=None, height=1000):
        self.shape = shape
        self.size = size
        self.position = position
        self.height = height
        if shape == 'circle':
            self.quat = np.array([1, 0.0, 0.0, 0]) # [w,x,y,z]
            self.pos_3d = torch.FloatTensor([position[0], position[1], 0])
            self.geom = fcl.Cylinder(size, height)
        elif shape == 'rect':
            self.quat = np.array([np.cos(rot/2), 0.0, 0.0, np.sin(rot/2)]) # [w,x,y,z]
            self.pos_3d = torch.FloatTensor([position[0], position[1], 0])
            self.geom = fcl.Box(size[0], size[1], height)

        self.cobj = fcl.CollisionObject(self.geom, fcl.Transform(self.quat, self.pos_3d))
        self.category = category