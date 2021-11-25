# After DiffCo Change: python setup.py install
#
# matplotlib tex 사용법:
# sudo apt-get install texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super
# pip install latex
# 

from active import *
from diffco.Obstacles import Obstacle
from diffco.DiffCo import DiffCo, vis
from diffco import vis
from time import time

def single_plot(robot, p, fig, link_plot, joint_plot, eff_plot, cfg_path_plots=None, path_history=None, save_dir=None, ax=None):
    points_traj = torch.cat([torch.zeros(len(p), 1, 2), robot.fkine(p)], dim=1)
    ends_alpha = 0.5
    lw = link_plot.get_lw()
    link_traj = [ax.plot(points[:, 0], points[:, 1], color='gray', alpha=ends_alpha, lw=lw, solid_capstyle='round')[0] for points in points_traj]
    joint_traj = [ax.plot(points[:-1, 0], points[:-1, 1], 'o', color='tab:red', alpha=ends_alpha, markersize=lw)[0] for points in points_traj]
    eff_traj = [ax.plot(points[-1:, 0], points[-1:, 1], 'o', color='black', alpha=ends_alpha, markersize=lw)[0] for points in points_traj]
    link_traj[0].set_color('green')
    offset = torch.FloatTensor([[0, 0], [0, 1], [-1, 1], [-1, 0]]) * np.pi*2
    for i, cfg_path in enumerate(cfg_path_plots):
        cfg_path.set_data(p[:, 0]+offset[i%4, 0], p[:, 1]+offset[i%4, 1])

def create_plots(robot, obstacles, dist_est, checker):
    from matplotlib.cm import get_cmap
    cmaps = [get_cmap('Reds'), get_cmap('Blues')]
    if robot.dof == 2:
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
        with sns.axes_style('ticks'): # 그리드가 없음
            for cat in range(num_class):
                c_ax = fig.add_subplot(gs[-1, cat])
                score = score_spline[:, :, cat] # estimated score grid
                c_ax.contourf(xx, yy, score, 8, cmap='coolwarm', vmin=-torch.abs(score).max(), vmax=torch.abs(score).max())
                contour = c_ax.contour(xx, yy, score, levels=8, linewidths=1, colors="k") 
                plt.clabel(contour, inline=1, fontsize=7)
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
    else:
        raise Exception("??")
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

def create_plots_gt(robot, obstacles, dist_est):
    from matplotlib.cm import get_cmap
    cmaps = [get_cmap('Reds'), get_cmap('Blues')]
    if robot.dof == 2:
        num_class = 1
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
        grid_points = grid_points.double()
        _, score_spline = dist_est(grid_points)
        score_spline = score_spline.reshape(size+[num_class]).double()
        c_axes = []
        with sns.axes_style('ticks'): # 그리드가 없음
            for cat in range(num_class):
                c_ax = fig.add_subplot(gs[-1, cat])
                score = score_spline[:, :, cat] # estimated score grid
                c_ax.contourf(xx, yy, score, 8, cmap='coolwarm', vmin=-torch.abs(score).max(), vmax=torch.abs(score).max())
                contour = c_ax.contour(xx, yy, score, levels=8, linewidths=1, colors="k") 
                plt.clabel(contour, inline=1, fontsize=7)
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
    else:
        raise Exception("??")
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

def create_plots_gt_3d(robot, obstacles, dist_est):
    size = [400, 400]
    yy, xx = torch.meshgrid(torch.linspace(-np.pi, np.pi, size[0]), torch.linspace(-np.pi, np.pi, size[1]))
    grid_points = torch.stack([xx, yy], dim=2).reshape((-1, 2))
    grid_points = grid_points.double()
    _, score_spline = dist_est(grid_points)
    score = score_spline.reshape(size).double()
    print("Score cal done.")
    # FCL 
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    surf1 = ax1.plot_surface(xx.numpy(),yy.numpy(),score.numpy(),rstride=2,cstride=2,cmap=plt.cm.coolwarm,linewidth=0.5,antialiased=True,vmin=np.nanmin(score),vmax=np.nanmax(score))
    ax1.set_xlabel('x axis')
    ax1.set_ylabel('y axis')
    ax1.set_zlabel('z axis')
    plt.colorbar(surf1,shrink=0.5,aspect=5)
    plt.title("FCL")
    plt.show()

def main(checking_method = 'diffco'):
    # Set envs
    DOF = 2
    env_name = '1rect_active'
    # Set hyper-paramters
    seed = 1918
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Get Data (Robot & Obstacles)
    dataset = torch.load('data/2d_{}dof_{}.pt'.format(DOF, env_name))
    cfgs = dataset['data'].double() # 각도
    labels = dataset['label'].reshape(-1, 1).double()
    dists = dataset['dist'].reshape(-1, 1).double() 
    obstacles = [list(o) for o in dataset['obs']]
    obj_obstacles = [Obstacle(*param) for param in obstacles] # 'rect', position: (-7, 3), size: (2, 2) # dataset['obs']: [('rect', (-7, 3), (2, 2))]
    robot = dataset['robot'](*dataset['rparam']) # link_length: 3.5, link_width: 0.3, DOF: 2 # dataset['rparam']: [3.5, 0.3, 2]
    #=================================================================================================================================
    # Get FCL Obj of Obstacles
    fcl_obs = [FCLObstacle(*param) for param in obstacles]
    fcl_collision_obj = [fobs.cobj for fobs in fcl_obs]
    # Robot Angles Set
    start_cfg = torch.zeros(robot.dof, dtype=cfgs.dtype) 
    target_cfg = torch.zeros(robot.dof, dtype=cfgs.dtype) 
    start_cfg[0] = np.pi/2 
    start_cfg[1] = -np.pi/6
    target_cfg[0] = 0 
    target_cfg[1] = np.pi/7
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
    train_num = 6000
    labels, dists = gt_checker.predict(cfgs[:train_num], distance=True)
    labels = labels.double()
    dists = dists.double()
    # Checking Method
    if checking_method == 'diffco':
        ### Train and Get DiffCo checker
        # Train DiffCo
        fkine = robot.fkine
        checker = DiffCo(obj_obstacles, kernel_func=kernel.FKKernel(fkine, kernel.RQKernel(10)), beta=1.0)
        """print("cfgs[:train_num].shape:", cfgs[:train_num].shape) # [6000, 2]
        print("labels[:train_num].shape:", labels[:train_num].shape) # 6000
        print("distance[:train_num].shape:", dists[:train_num].shape) # 6000"""
        checker.train(cfgs[:train_num], labels[:train_num].squeeze(), max_iteration=len(cfgs[:train_num]), distance=dists[:train_num].squeeze())
        print('Num of support points {}'.format(len(checker.support_points))) # [32, 2]; 개수는 자동으로 정해지는 듯
        # Get DiffCo
        fitting_target = 'dist'
        Epsilon = 0.01
        checker.fit_poly(kernel_func=kernel.Polyharmonic(1, Epsilon), target=fitting_target, fkine=fkine) 
        dist_est = checker.rbf_score # 위의 polyharmonic 함수
        print('MIN_SCORE = {:.6f}'.format(dist_est(cfgs[train_num:]).min()))
        """# Check DiffCo test ACC
        test_preds = (checker.score(cfgs[train_num:]) > 0) * 2 - 1
        test_acc = torch.sum(test_preds == labels[train_num:], dtype=torch.float32)/len(test_preds.view(-1))
        test_tpr = torch.sum(test_preds[labels[train_num:]==1] == 1, dtype=torch.float32) / len(test_preds[labels[train_num:]==1])
        test_tnr = torch.sum(test_preds[labels[train_num:]==-1] == -1, dtype=torch.float32) / len(test_preds[labels[train_num:]==-1])
        print('Test acc: {}, TPR {}, TNR {}'.format(test_acc, test_tpr, test_tnr))
        print(len(checker.gains), 'Support Points')"""
        # Create Plots
        fig, ax, link_plot, joint_plot, eff_plot, cfg_path_plots = create_plots(robot, obstacles, dist_est, checker)
    elif checking_method == 'fcl':
        fig, ax, link_plot, joint_plot, eff_plot, cfg_path_plots = create_plots_gt(robot, obstacles, gt_checker.predict)
    elif checking_method == 'fcl_3d':
        create_plots_gt_3d(robot, obstacles, gt_checker.predict)
        exit()
    else:
        pass
    # Single Plot and Save
    start_cfg_uq = start_cfg.unsqueeze(0)
    single_plot(robot, start_cfg_uq, fig, link_plot, joint_plot, eff_plot, cfg_path_plots=cfg_path_plots, ax=ax)
    fig_dir = 'figs/my_test/{random_seed}/{checking_method}'.format(random_seed=seed, checking_method=checking_method)
    if not isdir(fig_dir):
        makedirs(fig_dir)
    plt.savefig(join(fig_dir, '2d_{DOF}dof_{ename}_{checking_method}'.format(
        DOF=robot.dof, ename=env_name, checking_method=checking_method)), dpi=300)
    # Summary
    print('{} summary'.format(checking_method))

if __name__ == "__main__":
    # main(checking_method = 'diffco')
    # main(checking_method = 'fcl')
    main(checking_method='fcl_3d')
