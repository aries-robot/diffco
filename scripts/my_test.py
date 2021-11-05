from scripts.active import *

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
    main()