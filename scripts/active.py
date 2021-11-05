import sys
import json
from os.path import join, isdir, isfile
from os import makedirs
from diffco import DiffCo, MultiDiffCo, CollisionChecker
from diffco import kernel
from matplotlib import pyplot as plt
import numpy as np
import torch
from diffco.model import RevolutePlanarRobot
import fcl
from scipy import ndimage
from matplotlib import animation
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import seaborn as sns
sns.set()
import matplotlib.patheffects as path_effects
from diffco import utils
from diffco.Obstacles import FCLObstacle
from diffco.FCLChecker import FCLChecker
from speed_compare import gradient_free_traj_optimize, givengrad_traj_optimize

def traj_optimize(robot, dist_est, start_cfg, target_cfg, history=False):
    N_WAYPOINTS = 20
    NUM_RE_TRIALS = 5
    UPDATE_STEPS = 200
    dif_weight = 1
    max_move_weight = 10
    collision_weight = 10
    safety_margin = torch.FloatTensor([0]) #-3,  #-1 (shown)
    lr = 5e-1

    lowest_cost_solution = None
    lowest_cost = np.inf
    lowest_cost_trial = None
    lowest_cost_step = None
    best_valid_solution = None
    best_valid_cost = np.inf
    best_valid_step = None
    best_valid_trial = None
    
    trial_histories = []

    found = False
    for trial_time in range(NUM_RE_TRIALS):
        path_history = []
        if trial_time == 0 and False: # Temp
            init_path = torch.from_numpy(np.linspace(start_cfg, target_cfg, num=N_WAYPOINTS))
        else:
            init_path = (torch.rand(N_WAYPOINTS, robot.dof))*np.pi*2-np.pi
        init_path[0] = start_cfg
        init_path[-1] = target_cfg
        p = init_path.requires_grad_(True)
        opt = torch.optim.Adam([p], lr=lr)

        for step in range(UPDATE_STEPS):
            opt.zero_grad()
            collision_score = torch.clamp(dist_est(p)-safety_margin, min=0).sum()
            control_points = robot.fkine(p)
            max_move_cost = torch.clamp((control_points[1:, -1:]-control_points[:-1, -1:]).pow(2).sum(dim=2)-1**2, min=0).sum() \
                + torch.clamp((utils.wrap2pi(p[1:]-p[:-1])).pow(2).sum(dim=1)-(5*np.pi/180)**2, min=0).sum()
            diff = (control_points[1:]-control_points[:-1]).pow(2).sum()
            constraint_loss = collision_weight * collision_score + max_move_weight * max_move_cost
            objective_loss = dif_weight * diff
            loss = objective_loss + constraint_loss
            loss.backward()
            p.grad[[0, -1]] = 0.0
            opt.step()
            p.data = utils.wrap2pi(p.data)
            if history:
                path_history.append(p.data.clone())
            if constraint_loss.data.numpy() < lowest_cost:
                lowest_cost = loss.data.numpy()
                lowest_cost_solution = p.data.clone()
                lowest_cost_step = step
                lowest_cost_trial = trial_time
            if constraint_loss <= 1e-2:
                if objective_loss.data.numpy() < best_valid_cost:
                    best_valid_cost = objective_loss.data.numpy()
                    best_valid_solution = p.data.clone()
                    best_valid_step = step
                    best_valid_trial = trial_time
            if constraint_loss <= 1e-2 or step % (UPDATE_STEPS/5) == 0 or step == UPDATE_STEPS-1:
                print('Trial {}: Step {}, collision={:.3f}*{:.1f}, max_move={:.3f}*{:.1f}, diff={:.3f}*{:.1f}, Loss={:.3f}'.format(
                    trial_time, step, 
                    collision_score.item(), collision_weight,
                    max_move_cost.item(), max_move_weight,
                    diff.item(), dif_weight,
                    loss.item()))
        trial_histories.append(path_history)
        
        if best_valid_solution is not None:
            found = True
            break
    if not found:
        print('Did not find a valid solution after {} trials!\
            Giving the lowest cost solution'.format(NUM_RE_TRIALS))
        solution = lowest_cost_solution
        solution_step = lowest_cost_step
        solution_trial = lowest_cost_trial
    else:
        solution = best_valid_solution
        solution_step = best_valid_step
        solution_trial = best_valid_trial
    path_history = trial_histories[solution_trial] # Could be empty when history = false
    if not path_history:
        path_history.append(solution)
    else:
        path_history = path_history[:(solution_step+1)]
    return solution, path_history, solution_trial, solution_step # sum(trial_histories, []),

def create_plots(robot, obstacles, dist_est, checker):
    from matplotlib.cm import get_cmap
    cmaps = [get_cmap('Reds'), get_cmap('Blues')]

    if robot.dof > 2:
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111) 
    elif robot.dof == 2:
        # Show C-space at the same time
        num_class = getattr(checker, 'num_class', 1)

        fig = plt.figure(figsize=(3*(num_class), 3*(num_class+1)))
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"]})
        gs = fig.add_gridspec(num_class+1, num_class)
        ax = fig.add_subplot(gs[:-1, :])
        cfg_path_plots = []

        size = [400, 400]
        yy, xx = torch.meshgrid(torch.linspace(-np.pi, np.pi, size[0]), torch.linspace(-np.pi, np.pi, size[1]))
        grid_points = torch.stack([xx, yy], dim=2).reshape((-1, 2))
        grid_points = grid_points.double() if checker.support_points.dtype == torch.float64 else grid_points
        score_spline = dist_est(grid_points).reshape(size+[num_class])
        c_axes = []

        # 여기여기여기여기여기여기
        with sns.axes_style('ticks'): # 그리드가 없음
            for cat in range(num_class):
                c_ax = fig.add_subplot(gs[-1, cat])
                score = score_spline[:, :, cat] # estimated score grid
                color_mesh = c_ax.pcolormesh(xx, yy, score, cmap=cmaps[cat], vmin=-torch.abs(score).max(), vmax=torch.abs(score).max())
                c_support_points = checker.support_points[checker.gains[:, cat] != 0]
                c_ax.scatter(c_support_points[:, 0], c_support_points[:, 1], marker='.', c='black', s=1.5)
                c_ax.contour(xx, yy, score, levels=10, linewidths=1, alpha=0.4, ) #-1.5, -0.75, 0, 0.3
                for _ in range(4):
                    cfg_path, = c_ax.plot([], [], '-o', c='olivedrab', markersize=3)
                    cfg_path_plots.append(cfg_path)

                c_ax.set_aspect('equal', adjustable='box')
                c_ax.set_xlim(-np.pi, np.pi)
                c_ax.set_ylim(-np.pi, np.pi)
                c_ax.set_xticks([-np.pi, 0, np.pi])
                c_ax.set_xticklabels(['$-\pi$', '$0$', '$\pi$'])
                c_ax.set_yticks([-np.pi, 0, np.pi])
                c_ax.set_yticklabels(['$-\pi$', '$0$', '$\pi$'])


    # Plot ostacles
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([-4, 0, 4])
    ax.set_yticks([-4, 0, 4])
    for obs in obstacles:
        cat = obs[3] if len(obs) >= 4 else 1
        if obs[0] == 'circle':
            ax.add_patch(Circle(obs[1], obs[2], path_effects=[path_effects.withSimplePatchShadow()], color=cmaps[cat](0.5)))
        elif obs[0] == 'rect':
            ax.add_patch(Rectangle((obs[1][0]-float(obs[2][0])/2, obs[1][1]-float(obs[2][1])/2), obs[2][0], obs[2][1], path_effects=[path_effects.withSimplePatchShadow()], 
            color=cmaps[cat](0.5)))
    
    # Placeholder of the robot plot
    trans = ax.transData.transform
    lw = ((trans((1, robot.link_width))-trans((0,0)))*72/ax.figure.dpi)[1]
    link_plot, = ax.plot([], [], color='silver', alpha=0.1, lw=lw, solid_capstyle='round', path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()])
    joint_plot, = ax.plot([], [], 'o', color='tab:red', markersize=lw)
    eff_plot, = ax.plot([], [], 'o', color='black', markersize=lw)

    if robot.dof > 2:
        return fig, ax, link_plot, joint_plot, eff_plot
    elif robot.dof == 2:
        return fig, ax, link_plot, joint_plot, eff_plot, cfg_path_plots
    
def single_plot(robot, p, fig, link_plot, joint_plot, eff_plot, cfg_path_plots=None, path_history=None, save_dir=None, ax=None):
    points_traj = torch.cat([torch.zeros(len(p), 1, 2), robot.fkine(p)], dim=1)
    traj_alpha = 0.3
    ends_alpha = 0.5
    
    lw = link_plot.get_lw()
    link_traj = [ax.plot(points[:, 0], points[:, 1], color='gray', alpha=traj_alpha, lw=lw, solid_capstyle='round')[0] for points in points_traj]
    joint_traj = [ax.plot(points[:-1, 0], points[:-1, 1], 'o', color='tab:red', alpha=traj_alpha, markersize=lw)[0] for points in points_traj]
    eff_traj = [ax.plot(points[-1:, 0], points[-1:, 1], 'o', color='black', alpha=traj_alpha, markersize=lw)[0] for points in points_traj]
    for i in [0, -1]:
        link_traj[i].set_alpha(ends_alpha)
        link_traj[i].set_path_effects([path_effects.SimpleLineShadow(), path_effects.Normal()])
        joint_traj[i].set_alpha(ends_alpha)
        eff_traj[i].set_alpha(ends_alpha)
    link_traj[0].set_color('green')
    link_traj[-1].set_color('orange')

    offset = torch.FloatTensor([[0, 0], [0, 1], [-1, 1], [-1, 0]]) * np.pi*2
    for i, cfg_path in enumerate(cfg_path_plots):
        cfg_path.set_data(p[:, 0]+offset[i%4, 0], p[:, 1]+offset[i%4, 1])


def escape(robot, dist_est, start_cfg):
    N_WAYPOINTS = 20
    UPDATE_STEPS = 200

    safety_margin = -0.3 
    lr = 5e-2
    path_history = []
    init_path = start_cfg
    p = init_path.requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr)

    for step in range(N_WAYPOINTS):
        if step % 1 == 0:
            path_history.append(p.data.clone())
        opt.zero_grad()
        collision_score = dist_est(p)-safety_margin 
        loss = collision_score 
        loss.backward()
        opt.step()
        p.data = utils.wrap2pi(p.data)
        if collision_score <= 1e-4:
            break

    return torch.stack(path_history, dim=0)

def main(checking_method='diffco'):
    DOF = 2
    env_name = '1rect_active'

    dataset = torch.load('data/2d_{}dof_{}.pt'.format(DOF, env_name))
    cfgs = dataset['data'].double()
    labels = dataset['label'].reshape(-1, 1).double()
    dists = dataset['dist'].reshape(-1, 1).double() 
    obstacles = dataset['obs']
    obstacles = [list(o) for o in obstacles]
    robot = dataset['robot'](*dataset['rparam'])
    width = robot.link_width

    #=================================================================================================================================
    fcl_obs = [FCLObstacle(*param) for param in obstacles]
    fcl_collision_obj = [fobs.cobj for fobs in fcl_obs]

    label_type = 'binary'
    num_class = 1

    T = 11
    nu = 5 #5
    kai = 1500
    sigma = 0.3
    seed = 1918
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    num_init_points = 8000
    if label_type == 'binary':
        obs_managers = [fcl.DynamicAABBTreeCollisionManager()]
        obs_managers[0].registerObjects(fcl_collision_obj)
        obs_managers[0].setup()
    elif label_type == 'instance':
        obs_managers = [fcl.DynamicAABBTreeCollisionManager() for _ in fcl_obs]
        for mng, cobj in zip(obs_managers, fcl_collision_obj):
            mng.registerObjects([cobj])
    elif label_type == 'class':
        obs_managers = [fcl.DynamicAABBTreeCollisionManager() for _ in range(num_class)]
        obj_by_cls = [[] for _ in range(num_class)]
        for obj in fcl_obs:
            obj_by_cls[obj.category].append(obj.cobj)
        for mng, obj_group in zip(obs_managers, obj_by_cls):
            mng.registerObjects(obj_group)
    
    robot_links = robot.update_polygons(cfgs[0])
    robot_manager = fcl.DynamicAABBTreeCollisionManager()
    robot_manager.registerObjects(robot_links)
    robot_manager.setup()
    for mng in obs_managers:
        mng.setup()
    gt_checker = FCLChecker(obstacles, robot, robot_manager, obs_managers)

    train_num = 6000
    fkine = robot.fkine
    from time import time
    init_train_t = time()
    checker = MultiDiffCo(obstacles, kernel_func=kernel.FKKernel(fkine, kernel.RQKernel(10)), beta=1.0)
    labels, dists = gt_checker.predict(cfgs[:train_num], distance=True)
    labels = labels.double()
    dists = dists.double()
    checker.train(cfgs[:train_num], labels[:train_num], max_iteration=len(cfgs[:train_num]), distance=dists[:train_num])

    fitting_target = 'dist' 
    Epsilon = 0.01
    checker.fit_poly(kernel_func=kernel.Polyharmonic(1, Epsilon), target=fitting_target, fkine=fkine) 
    dist_est = checker.rbf_score # 위의 polyharmonic 함수
    init_train_t = time() - init_train_t
    print('MIN_SCORE = {:.6f}'.format(dist_est(cfgs[train_num:]).min()))
    
    positions = torch.FloatTensor(np.linspace(obstacles[0][1], [4, 3], T))
    start_cfg = torch.zeros(robot.dof, dtype=cfgs.dtype) 
    target_cfg = torch.zeros(robot.dof, dtype=cfgs.dtype) 
    start_cfg[0] = np.pi/2 
    start_cfg[1] = -np.pi/6
    target_cfg[0] = 0 
    target_cfg[1] = np.pi/7

    update_ts = []
    plan_ts = []
    for t, trans in zip(range(T), positions):
        ut = time()
        fcl_collision_obj[0].setTransform(fcl.Transform(
                    [trans[0], trans[1], 0]))
        for obs_mng in obs_managers:
            obs_mng.update()

        exploit_samples = torch.randn(nu, len(checker.gains), robot.dof, dtype=checker.support_points.dtype) * sigma + checker.support_points
        exploit_samples = utils.wrap2pi(exploit_samples).reshape(-1, robot.dof)

        explore_samples = torch.rand(kai, robot.dof, dtype=checker.support_points.dtype) * 2*np.pi - np.pi

        cfgs = torch.cat([exploit_samples, explore_samples, checker.support_points])
        labels, dists = gt_checker.predict(cfgs, distance=True)
        dists = dists.double()
        print('Collision {}, Free {}\n'.format((labels == 1).sum(), (labels==-1).sum()))

        gains = torch.cat([torch.zeros(len(exploit_samples)+len(explore_samples), checker.num_class, dtype=checker.gains.dtype), checker.gains])  
        added_hypothesis = checker.score(cfgs[:-len(checker.support_points)])
        hypothesis = torch.cat([added_hypothesis, checker.hypothesis]) 

        checker.train(cfgs, labels, gains=gains, hypothesis=hypothesis, distance=dists) 
        print('Num of support points {}'.format(len(checker.support_points)))
        checker.fit_poly(kernel_func=kernel.Polyharmonic(1, Epsilon), target=fitting_target, fkine=fkine, reg=0.1)
        update_ts.append(time()-ut)
        
        if checking_method == 'fcl':
            fcl_options = {
                'N_WAYPOINTS': 20,
                'NUM_RE_TRIALS': 5,
                'MAXITER': 200,
                'seed': seed,
                'history': False
            }
        elif checking_method == 'diffco':
            diffco_options = {
                'N_WAYPOINTS': 20,
                'NUM_RE_TRIALS': 5, 
                'MAXITER': 200,
                'safety_margin': -0.5, 
                'seed': seed,
                'history': False
            }

        print('t = {}'.format(t))
        if t % 1 == 0 and not torch.any(checker.predict(torch.stack([start_cfg, target_cfg], dim=0)) == 1):

            obstacles[0][1] = (trans[0], trans[1])
            cfg_path_plots = []
            if robot.dof > 2:
                fig, ax, link_plot, joint_plot, eff_plot = create_plots(robot, obstacles, dist_est, checker) # 플롯 생성
            elif robot.dof == 2:
                fig, ax, link_plot, joint_plot, eff_plot, cfg_path_plots = create_plots(robot, obstacles, dist_est, checker)
            
            ot = time()
            # Begin optimization==========
            if checking_method == 'diffco':
                solution_rec = givengrad_traj_optimize(
                    robot, dist_est, start_cfg, target_cfg, options=diffco_options)
                p = torch.FloatTensor(solution_rec['solution'])
            elif checking_method == 'fcl':
                solution_rec = gradient_free_traj_optimize(
                    robot, lambda cfg: gt_checker.predict(cfg, distance=False), start_cfg, target_cfg, options=fcl_options)
                p = torch.FloatTensor(solution_rec['solution'])
            # ============================
            plan_ts.append(time()-ot)

            path_dir = 'results/active_learning/path_2d_{}dof_{}_checker={}_seed{}_step{:02d}.json'.format(robot.dof, env_name, checking_method, seed, t)
            with open(path_dir, 'w+t') as f:
                json.dump(
                    {
                        'path': p.data.numpy().tolist(),
                    },
                    f, indent=1)
                print('Plan recorded in {}'.format(f.name))
            
            p = utils.make_continue(p)

            # single shot
            single_plot(robot, p, fig, link_plot, joint_plot, eff_plot, cfg_path_plots=cfg_path_plots, ax=ax)
            # plt.show()
            fig_dir = 'figs/active/{random_seed}/{checking_method}'.format(random_seed=seed, checking_method=checking_method)
            if not isdir(fig_dir):
                makedirs(fig_dir)
            plt.savefig(join(fig_dir, '2d_{DOF}dof_{ename}_{checking_method}_{step:02d}'.format(
                DOF=robot.dof, ename=env_name, checking_method=checking_method, step=t)), dpi=300)
    
    print('{} summary'.format(checking_method))
    print('Initial training {} sec.'.format(init_train_t))
    print('Update {} sec.'.format(update_ts))
    print('Planning {} sec.'.format(plan_ts))


if __name__ == "__main__":
    main(checking_method='diffco')
    main(checking_method='fcl')