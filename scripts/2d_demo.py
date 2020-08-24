import sys
sys.path.append('/home/yuheng/FastronPlus-pytorch/')
from Fastronpp import Fastron
from Fastronpp import kernel
from matplotlib import pyplot as plt
import numpy as np
import torch
from Fastronpp.model import RevolutePlanarRobot
import fcl
from scipy import ndimage
from matplotlib import animation
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import seaborn as sns
sns.set()
import matplotlib.patheffects as path_effects
from Fastronpp import utils
from Fastronpp.Obstacles import FCLObstacle


if __name__ == "__main__":
    # ========================== Data generqation =========================
    obstacles = [
        # ('circle', (3, 2), 2),
        # ('circle', (-2, 3), 1),
        # ('rect', (-2, 3), (1, 1)),
        ('rect', (1.7, 3), (2, 3)),
        ('rect', (-1.7, 3), (2, 3)),
        # ('rect', (0, -1), (10, 1)),
        # ('rect', (8, 7), 1),
        ]
    fcl_obs = [FCLObstacle(*param) for param in obstacles]
    env_name = 'narrow'

    DOF = 7
    width = 0.3
    link_length = 1
    robot = RevolutePlanarRobot(link_length, width, DOF) # (7, 1), (2, 2)

    np.random.seed(1917)
    torch.random.manual_seed(1917)
    num_init_points = 8000
    cfgs = 2*(torch.rand((num_init_points, DOF), dtype=torch.float32)-0.5) * np.pi
    labels = torch.zeros(num_init_points, dtype=torch.float)
    dists = torch.zeros(num_init_points, dtype=torch.float)
    
    robot_links = robot.update_polygons(cfgs[0])
    robot_manager = fcl.DynamicAABBTreeCollisionManager()
    obs_manager = fcl.DynamicAABBTreeCollisionManager()
    robot_manager.registerObjects(robot_links)
    obs_manager.registerObjects(fcl_obs)
    robot_manager.setup()
    obs_manager.setup()
    req = fcl.CollisionRequest(num_max_contacts=100, enable_contact=True)
    for i, cfg in enumerate(cfgs):
        robot.update_polygons(cfg)
        robot_manager.update()
        assert len(robot_manager.getObjects()) == DOF
        rdata = fcl.CollisionData(request = req)
        robot_manager.collide(obs_manager, rdata, fcl.defaultCollisionCallback)
        in_collision = rdata.result.is_collision
        labels[i] = 1 if in_collision else -1
        if in_collision:
            depths = torch.tensor([c.penetration_depth for c in rdata.result.contacts])
            dists[i] = depths.abs().max()
        else:
            ddata = fcl.DistanceData()
            robot_manager.distance(obs_manager, ddata, fcl.defaultDistanceCallback)
            dists[i] = -ddata.result.min_distance
    print('{} collisons, {} free'.format(torch.sum(labels==1), torch.sum(labels==-1)))
    dataset = {'data': cfgs, 'label': labels, 'dist': dists, 'obs': obstacles, 'robot': robot.__class__, 'rparam': [link_length, width, DOF, ]}
    torch.save(dataset, 'data/2d_{}dof_{}.pt'.format(DOF, env_name))
    # ========================== Data generqation =========================

    # DOF = 2
    # env_name = '1rect'

    # dataset = torch.load('data/2d_{}dof_{}.pt'.format(DOF, env_name))
    # cfgs = dataset['data']
    # labels = dataset['label']
    # dists = dataset['dist']
    # obstacles = dataset['obs']
    # robot = dataset['robot'](*dataset['rparam'])
    # width = robot.link_width
    # train_num = 3000
    # fkine = robot.fkine
    # checker = Fastron(obstacles, kernel_func=kernel.FKKernel(fkine, kernel.RQKernel(10)), beta=1.0)
    # checker.train(cfgs[:train_num], labels[:train_num], max_iteration=len(cfgs[:train_num]), distance=dists[:train_num])

    # # Check Fastron test ACC
    # test_preds = (checker.score(cfgs[train_num:]) > 0) * 2 - 1
    # test_acc = torch.sum(test_preds == labels[train_num:], dtype=torch.float32)/len(test_preds)
    # test_tpr = torch.sum(test_preds[labels[train_num:]==1] == 1, dtype=torch.float32) / len(test_preds[labels[train_num:]==1])
    # test_tnr = torch.sum(test_preds[labels[train_num:]==-1] == -1, dtype=torch.float32) / len(test_preds[labels[train_num:]==-1])
    # print('Test acc: {}, TPR {}, TNR {}'.format(test_acc, test_tpr, test_tnr))
    # assert(test_acc > 0.9)

    # fitting_target = 'label' # {label, dist, hypo}
    # Epsilon = 0.01
    # checker.fit_rbf(epsilon=Epsilon, target=fitting_target, fkine=fkine)
    # # checker.fit_poly(k=1, epsilon=Epsilon, target=fitting_target, fkine=fkine)
    # spline_func = checker.rbf_score
    # # spline_func = checker.poly_score

    # collision_cfgs = cfgs[labels==1]
    # # collision_cfgs = cfgs

    # if DOF == 7:
    #     fig = plt.figure(figsize=(8, 8))
    #     ax = fig.add_subplot(111) #, projection='3d'
    # elif DOF == 2:
    #     # Show C-space at the same time
    #     fig = plt.figure(figsize=(16, 8))
    #     ax = fig.add_subplot(121) #, projection='3d'
    #     c_ax = fig.add_subplot(122)
    #     size = [400, 400]
    #     yy, xx = torch.meshgrid(torch.linspace(-np.pi, np.pi, size[0]), torch.linspace(-np.pi, np.pi, size[1]))
    #     grid_points = torch.stack([xx, yy], axis=2).reshape((-1, 2))
    #     score_spline = spline_func(grid_points).reshape(size)
    #     score_fastron = checker.score(grid_points).reshape(size)
    #     score = (torch.sign(score_fastron)+1)/2*(score_spline-score_spline.min()) + (-torch.sign(score_fastron)+1)/2*(score_spline-score_spline.max())
    #     # score = score_spline
    #     c = c_ax.pcolormesh(xx, yy, score, cmap='RdBu_r', vmin=-torch.abs(score).max(), vmax=torch.abs(score).max())
    #     c_ax.scatter(checker.support_points[:, 0], checker.support_points[:, 1], marker='.', c='black')
    #     c_ax.contour(xx, yy, score, levels=0)
    #     c_ax.axis('equal')
    #     fig.colorbar(c, ax=c_ax)
    #     sparse_score = score[::10, ::10]
    #     score_grad_x = -ndimage.sobel(sparse_score.numpy(), axis=1)
    #     score_grad_y = -ndimage.sobel(sparse_score.numpy(), axis=0)
    #     score_grad = np.stack([score_grad_x, score_grad_y], axis=2)
    #     score_grad /= np.linalg.norm(score_grad, axis=2, keepdims=True)
    #     score_grad_x, score_grad_y = score_grad[:, :, 0], score_grad[:, :, 1]
    #     c_ax.quiver(xx[::10, ::10], yy[::10, ::10], score_grad_x, score_grad_y, color='red', width=2e-3, headwidth=2, headlength=5)
    #     cfg_point = Circle(collision_cfgs[0], radius=0.05, facecolor='orange', edgecolor='black', path_effects=[path_effects.withSimplePatchShadow()])
    #     c_ax.add_patch(cfg_point)


    # ax.axis('equal')
    # ax.set_xlim(-8, 7)
    # ax.set_ylim(-8, 7)
    # ax.set_aspect('equal', adjustable='box')
    # for obs in obstacles:
    #     if obs[0] == 'circle':
    #         ax.add_patch(Circle(obs[1], obs[2], path_effects=[path_effects.withSimplePatchShadow()]))
    #     elif obs[0] == 'rect':
    #         ax.add_patch(Rectangle((obs[1][0]-obs[2][0]/2, obs[1][1]-obs[2][1]/2), obs[2][0], obs[2][1], path_effects=[path_effects.withSimplePatchShadow()]))
    

    # trans = ax.transData.transform
    # lw = ((trans((1, width))-trans((0,0)))*72/ax.figure.dpi)[1]
    # link_plot, = ax.plot([], [], color='silver', lw=lw, path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()], solid_capstyle='round')
    # joint_plot, = ax.plot([], [], 'o', color='tab:red', markersize=lw)
    # eff_plot, = ax.plot([], [], 'o', color='black', markersize=lw)

    # global cur_cfg, cfg_cnt, opt, start_frame, cnt_down
    # lr = 5e-2
    # decay_weight = 1.0 # 0.1
    # decay_loss = torch.nn.MSELoss() #reduction='sum'
    # FPS = 15
    # pause_t = 0.5 # seconds
    # grad_clip = False
    # optimizer = torch.optim.Adam
    # def init():
    #     global cur_cfg, cfg_cnt, opt, start_frame, start_cfg, cnt_down
    #     cur_cfg = collision_cfgs[0].clone().detach().requires_grad_(True)
    #     start_cfg = collision_cfgs[0].clone().detach().requires_grad_(False)
    #     cfg_cnt = 0
    #     start_frame = 0
    #     cnt_down = int(pause_t*FPS)
    #     opt = optimizer([cur_cfg], lr=lr)
    #     if DOF==2:
    #         return link_plot, joint_plot, eff_plot, cfg_point
    #     else:
    #         return link_plot, joint_plot, eff_plot
    
    # def plot_robot(q):
    #     global cur_cfg, cfg_cnt, opt
    #     robot_points = robot.fkine(cur_cfg)[0]
    #     robot_points = torch.cat([torch.zeros(1, 2), robot_points])
    #     link_plot.set_data(robot_points[:, 0], robot_points[:, 1])
    #     joint_plot.set_data(robot_points[:-1, 0], robot_points[:-1, 1])
    #     eff_plot.set_data(robot_points[-1:, 0], robot_points[-1:, 1])

    #     return link_plot, joint_plot, eff_plot


    # def update(i):
    #     global cur_cfg, cfg_cnt, opt, start_frame, start_cfg, cnt_down
    #     with torch.no_grad():
    #         ret = plot_robot(cur_cfg)
    #         in_collision = checker.is_collision(cur_cfg)
    #     if DOF == 2:
    #         cfg_point.set_center(cur_cfg)
    #     opt.zero_grad()
    #     score = spline_func(cur_cfg)
    #     movement_loss = decay_weight * decay_loss(fkine(cur_cfg), fkine(start_cfg))
    #     loss = score + movement_loss
    #     loss.backward()
        
    #     print('CFG {} Frame {}: Score {:.8f}, movement_loss {:.8f}, grad {:.8f}'.format(
    #         cfg_cnt, i, score.data.numpy(), movement_loss.data.numpy(), cur_cfg.grad.norm().numpy()))
    #     # print(cur_cfg.grad.numpy())
    #     if  (score < -0.5 and not in_collision) or \
    #         (cur_cfg.grad.norm() < 10 and (i-start_frame) > 200):
    #         ax.set_title(('grad norm: {:.1f}, in collision: {}, score: {:.1f}, i - start_frame: {}'.format(
    #             cur_cfg.grad.norm(), 
    #             in_collision, 
    #             score,
    #             i-start_frame)))
    #         if not cnt_down:
    #             cfg_cnt += 1
    #             start_frame = i
    #             cur_cfg = collision_cfgs[cfg_cnt].clone().detach().requires_grad_(True)
    #             start_cfg = collision_cfgs[cfg_cnt].clone().detach().requires_grad_(False)
    #             opt = optimizer([cur_cfg], lr=lr)
    #             cnt_down = int(pause_t * FPS)
    #         else:
    #             cnt_down -= 1
    #     else:
    #         if grad_clip:
    #             # torch.nn.utils.clip_grad(cur_cfg, 20)
    #             cur_cfg.grad.clamp_(-2, 2)
    #         opt.step()
    #         cur_cfg.data = utils.wrap2pi(cur_cfg.data)
    #         ax.set_title('Configuration {}, Score {:.1f}, Collision {}'.format(cfg_cnt, score.item(), in_collision))
    #     if DOF == 2:
    #         return ret+(cfg_point, )
    #     else:
    #         return ret
    
    # ani = animation.FuncAnimation(fig, update, frames=900, interval=1, blit=False, init_func=init)
    # plt.show()
    # ani.save('results/2d_{}dof_{}_gradclip_{}_fitting_{}_decay_{}_eps_{}.mp4'.format(
    #     DOF, env_name, grad_clip, fitting_target, decay_weight, Epsilon), fps=FPS)


