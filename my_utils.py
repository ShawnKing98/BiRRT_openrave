import numpy as np
import time
from openravepy import *


def cal_variance(points):
    '''
    Calculate the variance of the 3 dimensions of C-space respectively
    :param points: a list of np array
    :return vars: a tuple contains the variance in 3 dimensions
    '''
    theta_weight = 1/np.pi
    x_var, y_var, theta_var = np.var(points, axis=0)
    theta_var = theta_var * theta_weight ** 2
    return x_var, y_var, theta_var


def cal_distance(q1, q2):
    '''
    Calculate the Euclidean distance with weight between two configuration
    :param q1, q2: two configurations
    :return distance: distance between two configurations
    '''
    x1, y1, theta1 = q1
    x2, y2, theta2 = q2
    dx = x1 - x2
    dy = y1 - y2
    dtheta = theta1 - theta2
    while dtheta > np.pi:
        dtheta -= 2 * np.pi
    while dtheta < -np.pi:
        dtheta += 2 * np.pi
    dtheta *= 1/np.pi   # weight on the dtheta term
    return np.linalg.norm([dx, dy, dtheta])

def waitrobot(robot):
    """busy wait for robot completion"""
    while not robot.GetController().IsDone():
        time.sleep(0.01)


def ConvertPathToTrajectory(env, robot, path=[]):
    '''
    Given a list of C-space nodes, convert it to an openrave trajectory that can be executed by the robot controller.
    :param env: an openrave environment
    :param robot: an openrave robot
    :param path: a list of C-space nodes
    :return traj: an openrave trajectory
    '''
    if not path:
        return None
    # init the configuration of trajectory
    traj = RaveCreateTrajectory(env, '')
    traj_config = robot.GetActiveConfigurationSpecification()
    traj_config.AddDerivativeGroups(1, False)
    traj_config.AddDeltaTimeGroup()
    traj.Init(traj_config)
    # insert passing point
    for i in range(len(path)):
        traj.Insert(i, np.concatenate([path[i].q, path[i].dq, path[i].dt]))
    # interpolation
    planningutils.RetimeAffineTrajectory(traj, maxvelocities=np.ones(3), maxaccelerations=5 * np.ones(3), hastimestamps=True)
    return traj
