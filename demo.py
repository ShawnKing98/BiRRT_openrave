import numpy as np
import time
from openravepy import *
from env_config import construct_env
# from kinodynamic_rrt import RRT
from kinodynamic_birrt import RRT
from my_utils import ConvertPathToTrajectory, waitrobot

if __name__ == '__main__':
    print "Shawn's EECS 498 final project demo: Kinodynamic BiRRT"
    visualization = raw_input("Do you wish to turn on the searching process visualization? (y/n)\n")
    if visualization == 'y':
        visualization = True
    elif visualization == 'n':
        visualization = False
    else:
        raise ValueError("Invalid visualization argument input:", visualization)
    seed = raw_input("Please input the random seed. I personally recommend 0 or 2.\n")
    seed = int(seed)
    print "Loading the environment. There's a small chance the environment not loaded properly, if you can only see a robot without environment, please just restart this program."
    env, robot = construct_env()
    time.sleep(0.1)
    start_config = [-3, -1.2, 0]
    goal_config = [3.2, 0, 0]
    robot.SetActiveDOFValues(np.array(start_config))

    # start solving
    print "--------------------------------------------------------------------------------------------------------------------------"
    print "Start searching for path. The maximum running time is set to 20 minutes, but normally it would take less than 10 minutes."
    print "--------------------------------------------------------------------------------------------------------------------------"
    rrt_solver = RRT(env, robot, start_config, goal_config, primitive_num=3, max_time=20*60, random_seed=seed)
    time.sleep(0.1)
    with env:
        path, runtime = rrt_solver.run(visualization=True)
        robot.SetActiveDOFValues(np.array(start_config))
        rrt_solver.handles = []
        path = rrt_solver.shortcut_smoothing(path)
        traj = ConvertPathToTrajectory(env, robot, path)
        if traj is not None:
            rrt_solver.draw_path(path)
    if traj is not None:
        raw_input("Press any key to execute the trajectory")
        print "Trajectory to be executed in 3 seconds..."
        time.sleep(3)
        robot.GetController().SetPath(traj)
        waitrobot(robot)
    raw_input("Press any key to exit")
