import time
import numpy as np
from openravepy import *



def construct_env():
    '''
    Construct an enviroment by loading an xml file. Create a robot from scratch named "Wand".
    :return env: the WanderMaze environment
    :return robot: the Wand robot
    '''
    env = Environment()
    env.SetViewer('qtcoin')
    collisionChecker = RaveCreateCollisionChecker(env, 'ode')
    env.SetCollisionChecker(collisionChecker)
    env.Reset()
    time.sleep(0.1)
    # env.Load('WanderMaze.env.xml')
    env.Load('WanderMaze_16_block.env.xml')
    time.sleep(0.1)
    # create a Wand robot from scratch
    robot = RaveCreateRobot(env, '')
    robot.SetName('Wand')
    robot.InitFromBoxes(np.array([[0, 0, 0.05, 0.5, 0.05, 0.05], [-0.4, 0, 0.12, 0.05, 0.05, 0.07]]), True)
    controller = RaveCreateController(env, 'IdealController')
    robot.SetController(controller)
    time.sleep(0.1)
    env.Add(robot, True)
    robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])

    time.sleep(0.1)
    return env, robot

if __name__ == '__main__':
    env, robot = construct_env()
    start_config = [-3, -1.2, 0]
    goal_config = [3.2, 0, 0]
    robot.SetActiveDOFValues(np.array(goal_config))
