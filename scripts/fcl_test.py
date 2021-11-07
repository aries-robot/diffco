import fcl
import numpy as np

v1 = np.array([1.0, 2.0, 3.0])
v2 = np.array([2.0, 1.0, 3.0])
v3 = np.array([3.0, 2.0, 1.0])
x, y, z = 1, 2, 3
rad, lz = 1.0, 3.0
n = np.array([1.0, 0.0, 0.0])
d = 5.0

box = fcl.Box(x, y, z)          # Axis-aligned box with given side lengths
sphere = fcl.Sphere(rad)           # Sphere with given radius
cone = fcl.Cone(rad, lz)         # Cone with given radius and cylinder height along z-axis
verts = np.array([[1.0, 1.0, 1.0],
                  [2.0, 1.0, 1.0],
                  [1.0, 2.0, 1.0],
                  [1.0, 1.0, 2.0]])
tris  = np.array([[0,2,1],
                  [0,3,2],
                  [0,1,3],
                  [1,2,3]])
mesh = fcl.BVHModel()
mesh.beginModel(len(verts), len(tris))
mesh.addSubModel(verts, tris)
mesh.endModel()

objs1 = [fcl.CollisionObject(box), fcl.CollisionObject(sphere), fcl.CollisionObject(cone)]
objs2 = [fcl.CollisionObject(cone), fcl.CollisionObject(mesh)]

manager1 = fcl.DynamicAABBTreeCollisionManager()
manager2 = fcl.DynamicAABBTreeCollisionManager()

manager1.registerObjects(objs1)
manager2.registerObjects(objs2)

manager1.setup()
manager2.setup()

##=====================================================================
## Managed internal (sub-n^2) distance checking
##=====================================================================
ddata = fcl.DistanceData()
manager1.distance(ddata, fcl.defaultDistanceCallback)
print(ddata.result.min_distance)