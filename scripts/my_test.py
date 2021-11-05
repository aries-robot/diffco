# After DiffCo Change: python setup.py install
from active import *
from diffco.Obstacles import Obstacle
from diffco.DiffCo import DiffCo, vis
from diffco import vis
from time import time

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

def main():
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
    obj_obstacles = [Obstacle(*param) for param in obstacles]
    robot = dataset['robot'](*dataset['rparam'])

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

    #=====================================================================================================
    """# Check DiffCo test ACC
    test_preds = (checker.score(cfgs[train_num:]) > 0) * 2 - 1
    test_acc = torch.sum(test_preds == labels[train_num:], dtype=torch.float32)/len(test_preds.view(-1))
    test_tpr = torch.sum(test_preds[labels[train_num:]==1] == 1, dtype=torch.float32) / len(test_preds[labels[train_num:]==1])
    test_tnr = torch.sum(test_preds[labels[train_num:]==-1] == -1, dtype=torch.float32) / len(test_preds[labels[train_num:]==-1])
    print('Test acc: {}, TPR {}, TNR {}'.format(test_acc, test_tpr, test_tnr))
    print(len(checker.gains), 'Support Points')"""
    # Optimization Method
    checking_method = 'diffco'
    if checking_method == 'diffco':
        diffco_options = {
            'N_WAYPOINTS': 20,
            'NUM_RE_TRIALS': 5, 
            'MAXITER': 200,
            'safety_margin': -0.1, 
            'seed': seed,
            'history': False
        }
    # Check Collision in start or end
    if torch.any(checker.predict(torch.stack([start_cfg, target_cfg], dim=0)) == 1):
        raise Exception("Collision in start or target.")
    # Create Plots
    fig, ax, link_plot, joint_plot, eff_plot, cfg_path_plots = create_plots(robot, obstacles, dist_est, checker)
    """# Optimization
    ot = time()
    if checking_method == 'diffco':
        solution_rec = givengrad_traj_optimize(
            robot, dist_est, start_cfg, target_cfg, options=diffco_options)
        p = torch.FloatTensor(solution_rec['solution'])
    plan_ts = time()-ot
    # Make Continuous (-pi and pi)
    p = utils.make_continue(p)"""
    ################################################# p가 뭔지 확인해야 함!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Single Plot and Save
    single_plot(robot, p, fig, link_plot, joint_plot, eff_plot, cfg_path_plots=cfg_path_plots, ax=ax)
    fig_dir = 'figs/my_test/{random_seed}/{checking_method}'.format(random_seed=seed, checking_method=checking_method)
    if not isdir(fig_dir):
        makedirs(fig_dir)
    plt.savefig(join(fig_dir, '2d_{DOF}dof_{ename}_{checking_method}'.format(
        DOF=robot.dof, ename=env_name, checking_method=checking_method)), dpi=300)
    # Summary
    print('{} summary'.format(checking_method))

if __name__ == "__main__":
    main()
