from active import *
from diffco.Obstacles import Obstacle
from time import time

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
    obstacles = dataset['obs']
    raw_obstacles = [list(o) for o in obstacles]
    obj_obstacles = [Obstacle(*param) for param in obstacles]
    robot = dataset['robot'](*dataset['rparam'])

    #=================================================================================================================================
    # Get FCL Obj of Obstacles
    fcl_obs = [FCLObstacle(*param) for param in raw_obstacles]
    fcl_collision_obj = [fobs.cobj for fobs in fcl_obs]
    # Robot Angles Set
    position = torch.FloatTensor([-1.5000,  3.0000])
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
    fcl_collision_obj[0].setTransform(fcl.Transform([position[0], position[1], 0]))
    for obs_mng in obs_managers:
        obs_mng.update()
    # Get FCL checker
    gt_checker = FCLChecker(obj_obstacles, robot, robot_manager, obs_managers)
    train_num = 6000
    labels, dists = gt_checker.predict(cfgs[:train_num], distance=True)
    labels = labels.double()
    dists = dists.double()
    # Train and Get DiffCo checker
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
    # Timing Set
    plan_ts = []
    ut = time() # Time Start
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
            'safety_margin': -0.5, 
            'seed': seed,
            'history': False
        }
    # Check Collision in start or end
    if torch.any(checker.predict(torch.stack([start_cfg, target_cfg], dim=0)) == 1):
        raise Exception("Collision in start or target.")
    # 
    obstacles[0][1] = (position[0], position[1])
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
    plt.savefig(join(fig_dir, '2d_{DOF}dof_{ename}_{checking_method}'.format(
        DOF=robot.dof, ename=env_name, checking_method=checking_method)), dpi=300)
    
    print('{} summary'.format(checking_method))
    print('Initial training {} sec.'.format(init_train_t))
    print('Update {} sec.'.format(update_ts))
    print('Planning {} sec.'.format(plan_ts))












if __name__ == "__main__":
    main()