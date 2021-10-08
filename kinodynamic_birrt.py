import time
import numpy as np
from openravepy import *
from kd_tree import Node, Tree
from my_utils import waitrobot, cal_distance

class RRT(object):
    def __init__(self, env, robot, start_config, goal_config, primitive_num=3, sample_method1='uniform', sample_method2='bridge', max_time=10*60, random_seed=None):
        self.env = env
        self.robot = robot
        self.start_config = np.array(start_config)
        self.goal_config = np.array(goal_config)
        self.criterion = np.array([0.05, 0.05, np.pi/18])
        self.max_time = max_time    # second as unit
        self.handles = []    # used for drawing
        self.dt = 0.1
        self.q_min = np.array([-4, -2, -np.pi])
        self.q_max = np.array([4, 2, np.pi])
        self.v_min = np.array([-1, -1, -np.pi/5])
        self.v_max = np.array([1, 1, np.pi/5])
        self.a_min = np.array([-5, -5, -np.pi])
        self.a_max = np.array([5, 5, np.pi])
        self.primitive_num = primitive_num
        self.sample_method1 = sample_method1
        self.sample_method2 = sample_method2
        self.delta_a = (self.a_max-self.a_min)/(self.primitive_num-1)
        start_node = Node(0, np.array(start_config), np.zeros(3), np.array([0]))
        self.kd_tree1 = Tree(start_node, [])
        end_node = Node(0, np.array(goal_config), np.zeros(3), np.array([0]))
        self.kd_tree2 = Tree(end_node, [])
        if random_seed is not None:
            np.random.seed(random_seed)

    def reset(self):
        self.handles = []
        start_node = Node(0, np.array(self.start_config), np.zeros(3), np.array([0]))
        self.kd_tree1 = Tree(start_node, [])
        end_node = Node(0, np.array(self.goal_config), np.zeros(3), np.array([0]))
        self.kd_tree2 = Tree(end_node, [])

    def clip(self, value, attribute='q'):
        value_new = value.copy()
        if attribute == 'q':
            for i in range(2):
                if value_new[i] < self.q_min[i]:
                    value_new[i] = self.q_min[i]
                if value_new[i] > self.q_max[i]:
                    value_new[i] = self.q_max[i]
            while value_new[2] < -np.pi:
                value_new[2] = value_new[2] + 2*np.pi
            while value_new[2] > np.pi:
                value_new[2] = value_new[2] - 2*np.pi
        elif attribute == 'v':
            for i in range(3):
                if value_new[i] < self.v_min[i]:
                    value_new[i] = self.v_min[i]
                if value_new[i] > self.v_max[i]:
                    value_new[i] = self.v_max[i]
        return value_new

    def draw_robot(self, q):
        self.robot.SetActiveDOFValues(q)
        self.handles.append(self.env.drawtrimesh(points=np.array(((0.04, 0, 0.05), (-0.02, 0.02, 0.05), (0, 0, 0.05))),
                                       indices=None,
                                       colors=np.array([0, 0, 1])))
        self.handles.append(self.env.drawtrimesh(points=np.array(((0.04, 0, 0.05), (-0.02, -0.02, 0.05), (0, 0, 0.05))),
                                                 indices=None,
                                                 colors=np.array([0, 0, 1])))
        self.handles[-1].SetTransform(self.robot.GetTransform())
        self.handles[-2].SetTransform(self.robot.GetTransform())

    # extend one step toward the target, return the newly added node
    def one_step_extend(self, tree_id, target, nearest_node=None, visualization=False, allow_away=False):
        if tree_id == 1:
            kd_tree = self.kd_tree1
            line_color = np.array([[1, 0, 0], [1, 0, 0]])
        elif tree_id == 2:
            kd_tree = self.kd_tree2
            line_color = np.array([[0, 1, 1], [0, 1, 1]])
        else:
            raise ValueError('Invalid tree id:', tree_id)
        if nearest_node is None:
            nearest_node, min_dist = kd_tree.find_nearest_node(*target)
            assert nearest_node in kd_tree.allnodes
        else:
            min_dist = cal_distance(nearest_node.q, target)
        q_near, v_near = nearest_node.q, nearest_node.dq
        q_new, v_new = None, None
        # search for the best primitive
        if allow_away:
            min_dist = np.inf       # allow the robot to drive away from target
        for i in range(self.primitive_num):
            for j in range(self.primitive_num):
                for k in range(self.primitive_num):
                    a = self.a_min + np.array([i, j, k])*self.delta_a
                    v = v_near + a*self.dt
                    v = self.clip(v, 'v')
                    q = q_near + 0.5*(v_near+v)*self.dt
                    dist = cal_distance(q, target)
                    if dist < min_dist:
                        # check collision twice
                        self.robot.SetActiveDOFValues(q)
                        if self.env.CheckCollision(self.robot):
                            continue
                        self.robot.SetActiveDOFValues((q+q_near)/2)
                        if self.env.CheckCollision(self.robot):
                            continue
                        min_dist = dist
                        q_new = q
                        v_new = v
        if q_new is None:
            node_new = None
        else:
            kd_tree.insert(q_new, v_new, np.array([self.dt]))
            node_new = kd_tree.allnodes[-1]
            node_new.parent = nearest_node
            if visualization:
                point1 = [q_near[0], q_near[1], 0.05]
                point2 = [q_new[0], q_new[1], 0.05]
                self.draw_robot(q_new)
                self.handles.append(self.env.drawlinestrip(points=np.array([point1, point2]),
                                                 linewidth=0.02,
                                                 colors=line_color))
        return node_new

    def multi_steps_extend(self, tree_id, target, visualization=False, onestep=False):
        if tree_id == 1:
            node, _ = self.kd_tree1.find_nearest_node(*target)
        elif tree_id == 2:
            node, _ = self.kd_tree2.find_nearest_node(*target)
        if visualization:
            target_handle_index = len(self.handles)
            self.handles.append(self.env.plot3(points=np.array([target[0], target[1], 0.05]),
                                               pointsize=0.05,
                                               colors=[1, 1, 0],
                                               drawstyle=1))
            near_handle_index = len(self.handles)
            self.handles.append(self.env.plot3(points=np.array([node.q[0], node.q[1], 0.05]),
                                               pointsize=0.05,
                                               colors=[0, 1, 0],
                                               drawstyle=1))
            line_handle_index = len(self.handles)
            self.handles.append(self.env.drawlinestrip(points=np.array([[target[0], target[1], 0.05], [node.q[0], node.q[1], 0.05]]),
                                                       linewidth=0.1,
                                                       colors=np.array([1, 1, 1])))

        result = None
        while True:
            node = self.one_step_extend(tree_id, target, node, visualization, allow_away=onestep)
            if onestep:
                result = False
            elif node is None:
                result = False  # failed to reach the target
            elif (target-self.criterion < node.q).all() and (node.q < target+self.criterion).all():
                result = True   # succeeded to reach the target
            if result is not None:
                if visualization:
                    self.handles.pop(line_handle_index)
                    self.handles.pop(near_handle_index)
                    self.handles.pop(target_handle_index)
                return result

    def bridge_sample(self, std=np.array([1, 1, np.pi/2])):
        while True:
            q1 = np.random.uniform(self.q_min, self.q_max)
            self.robot.SetActiveDOFValues(q1)
            if not self.env.CheckCollision(self.robot):
                continue
            q2 = np.random.normal(q1, std)
            q2 = self.clip(q2)
            self.robot.SetActiveDOFValues(q2)
            if not self.env.CheckCollision(self.robot):
                continue
            q_middle = (q1+q2)/2
            self.robot.SetActiveDOFValues(q_middle)
            if not self.env.CheckCollision(self.robot):
                return q_middle

    def run(self, visualization=False):
        start_time = time.time()
        path = []
        tree_id = 1
        q1_old, q1_new, q2_old, q2_new = None, None, None, None
        while time.time()-start_time < self.max_time:
            # if self.kd_tree1.size <= self.kd_tree2.size:
            #     tree_id = 1
            # else:
            #     tree_id = 2
            if tree_id == 1:
                # sample a point and do multi-steps extend
                if self.sample_method1 == 'uniform':
                    q_rand = np.random.uniform(self.q_min, self.q_max)
                elif self.sample_method1 == 'bridge':
                    q_rand = self.bridge_sample()
                else:
                    raise ValueError('Invalid sample method:', self.sample_method1)
                q1_old = self.kd_tree1.allnodes[-1].q
                self.multi_steps_extend(1, q_rand, visualization)
                q1_new = self.kd_tree1.allnodes[-1].q
                # if this tree got extended, let the other tree do multi-steps extend towards the newly added node and enter the other tree's turn
                if (q1_new != q1_old).any():
                # if True:
                    q2_old = self.kd_tree2.allnodes[-1].q
                    result = self.multi_steps_extend(2, q1_new, visualization, onestep=True)
                    q2_new = self.kd_tree2.allnodes[-1].q
                    tree_id = 2
                # if this tree didn't get extended, let the other tree do multi-steps extend towards the newly added point
                elif (q2_old != q2_new).any():
                    q2_old = self.kd_tree2.allnodes[-1].q
                    result = self.multi_steps_extend(2, q1_new, visualization, onestep=True)
                    q2_new = self.kd_tree2.allnodes[-1].q
                    # if (q2_new != q2_old).any():
                    tree_id = 2
                if result:
                    print 'Solution found!'
                    path = self.get_path()
                    runtime = time.time() - start_time
                    break

            elif tree_id == 2:
                # sample a point and do multi-steps extend
                if self.sample_method2 == 'uniform':
                    q_rand = np.random.uniform(self.q_min, self.q_max)
                elif self.sample_method2 == 'bridge':
                    q_rand = self.bridge_sample()
                else:
                    raise ValueError('Invalid sample method:', self.sample_method2)
                q2_old = self.kd_tree2.allnodes[-1].q
                self.multi_steps_extend(2, q_rand, visualization, onestep=True)
                q2_new = self.kd_tree2.allnodes[-1].q
                # if this tree got extended, let the other tree do multi-steps extend towards the newly added node and enter the other tree's turn
                if (q2_new != q2_old).any():
                # if True:
                    q1_old = self.kd_tree1.allnodes[-1].q
                    result = self.multi_steps_extend(1, q2_new, visualization)
                    q1_new = self.kd_tree1.allnodes[-1].q
                    tree_id = 1
                # if this tree didn't get extended, let the other tree do multi-steps extend towards the newly added point
                elif (q1_new != q1_old).any():
                    q1_old = self.kd_tree1.allnodes[-1].q
                    result = self.multi_steps_extend(1, q2_new, visualization)
                    q1_new = self.kd_tree1.allnodes[-1].q
                    tree_id = 1
                if result:
                    print 'Solution found!'
                    path = self.get_path()
                    runtime = time.time() - start_time
                    break
                    # if (q1_new != q1_old).any():
                    #     tree_id = 1
        if path == []:
            print "Run timeout!"
            runtime = time.time() - start_time
        print "Total runtime in second:", runtime
        return path, runtime

    def get_path(self):
        node = self.kd_tree1.allnodes[-1]
        path = [node]
        while node.parent is not None:
            node = node.parent
            path.append(node)
        path.reverse()
        node = self.kd_tree2.allnodes[-1]
        path.append(node)
        while node.parent is not None:
            node = node.parent
            path.append(node)
        return path

    def shortcut_smoothing(self, path, total_iter=500):
        if path == []:
            return None
        print "Path length before smoothing:", len(path)
        print "Smoothing path..."
        for iter in range(total_iter):
            start_index = np.random.randint(0, len(path))
            end_index = np.random.randint(0, len(path))
            while abs(start_index - end_index) <= 1:
                end_index = np.random.randint(0, len(path))
            if start_index > end_index:
                start_index, end_index = end_index, start_index
            new_path = path[: start_index+1]
            node = path[start_index]
            target = path[end_index].q

            reach_flag = True
            while not ((target-self.criterion < node.q).all() and (node.q < target+self.criterion).all()):
                min_dist = cal_distance(node.q, target)
                q_new, v_new = None, None
                for i in range(self.primitive_num):
                    for j in range(self.primitive_num):
                        for k in range(self.primitive_num):
                            a = self.a_min + np.array([i, j, k]) * self.delta_a
                            v = node.dq + a * self.dt
                            v = self.clip(v, 'v')
                            q = node.q + 0.5 * (node.dq + v) * self.dt
                            dist = cal_distance(q, target)
                            if dist < min_dist:
                                # check collision twice
                                self.robot.SetActiveDOFValues(q)
                                if self.env.CheckCollision(self.robot):
                                    continue
                                self.robot.SetActiveDOFValues((q + node.q) / 2)
                                if self.env.CheckCollision(self.robot):
                                    continue
                                min_dist = dist
                                q_new = q
                                v_new = v
                if q_new is None:
                    reach_flag = False
                    break
                else:
                    node = Node(None, q_new, v_new, np.array([self.dt]), node)
                    new_path.append(node)
            if reach_flag:
                for index in range(end_index, len(path)):
                    new_path.append(path[index])
                path = new_path
        print "Path length after smoothing:", len(path)
        return path

    def draw_path(self, path):
        points = []
        self.handles = []
        with self.env:
            for node in path:
                points.append([node.q[0], node.q[1], 0.05])
                self.draw_robot(node.q)
        self.handles.append(self.env.drawlinestrip(points=np.array(points),
                                         linewidth=0.1,
                                         colors=np.array([0, 0, 0])))
