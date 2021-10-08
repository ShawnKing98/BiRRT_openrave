from openravepy import *
import numpy as np
from my_utils import cal_variance, cal_distance


# define a basic node class
class Node:
    def __init__(self, id_in, q_in, dq_in, dt_in, parent_in=None, tree_in=None):
        self.id = id_in
        self.q = q_in       # a 3-element np array
        self.dq = dq_in     # a 3-element np array
        self.dt = dt_in       # a 1-element np array
        self.parent = parent_in
        self.tree = tree_in

    def printme(self):
        if self.parent is None:
            parentid = 'None'
        else:
            parentid = self.parent.id
        if self.tree is None or self.tree.split is None:
            tree_split = 'None'
        else:
            tree_split = self.tree.split
        print "Node id:", self.id, "| parentid:", parentid, "| q =", self.q, "| dq =", self.dq, "| tree split:", tree_split

# define a kd_tree
class Tree:
    def __init__(self, root, allnodes, new_node=True, nodes=None, split=None, parent=None):
        self.root = root
        self.root.tree = self
        self.allnodes = allnodes
        if new_node:
            self.allnodes.append(self.root)
        if nodes is None:
            self.nodes = [self.root]
            self.size = 1
        else:
            self.nodes = nodes
            self.size = len(nodes)
        self.split = split      # which dimension to split the data
        self.parent = parent    # parent tree
        self.left_child = None  # child tree
        self.right_child = None

    def printme(self):
        if self.parent is None:
            parentid = 'None'
        else:
            parentid = self.parent.root.id
        if self.split is None:
            split = 'None'
        else:
            split = self.split
        print "Node id:", self.root.id, "| parentid:", parentid, "| q =", self.root.q, "| tree split:", split
        if self.left_child is not None:
            self.left_child.printme()
        if self.right_child is not None:
            self.right_child.printme()

    # given a configuration, find the leaf tree it should be attached to in current tree structure
    def locate_node(self, *value):
        if self.split is None:
            assert self.size == 1
            return self
        if value[self.split] < self.root.q[self.split]:
            if self.left_child is None:
                return self
            else:
                return self.left_child.locate_node(*value)
        else:
            if self.right_child is None:
                return self
            else:
                return self.right_child.locate_node(*value)

    # given a configuration, find the nearest node and the distance between them
    def find_nearest_node(self, *value):
        tree = self.locate_node(*value)
        nearest_node = None
        min_dist = np.inf
        for self_node in tree.nodes:
            dist = cal_distance(value, self_node.q)
            if min_dist > dist:
                min_dist = dist
                nearest_node = self_node
        while tree.parent is not None:
            parent = tree.parent
            q_pedal = parent.root.q.copy()
            for i in range(len(q_pedal)):
                if i != parent.split:
                    q_pedal[i] = value[i]
            # check if we need to search the brother tree
            if cal_distance(value, q_pedal) < min_dist:
                if tree == parent.left_child:
                    brother = parent.right_child
                else:
                    brother = parent.left_child
                if brother is not None:
                    for brother_node in brother.nodes:
                        dist = cal_distance(value, brother_node.q)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_node = brother_node
            tree = parent
        return nearest_node, min_dist

    # return the tree itself
    def insert(self, *value):
        self.size += 1
        node = Node(len(self.allnodes), *value)
        if self.split is None:
            assert self.size == 2
            var_list = cal_variance([self.root.q, node.q])
            max_var = 0
            for i in range(len(var_list)):
                if var_list[i] >= max_var:
                    max_var = var_list[i]
                    self.split = i
        # tree is the left/right child tree that has been inserted
        if node.q[self.split] < self.root.q[self.split]:
            if self.left_child is None:
                tree = Tree(node, self.allnodes, parent=self)
            else:
                tree = self.left_child.insert(*value)
            self.left_child = tree
        else:
            if self.right_child is None:
                tree = Tree(node, self.allnodes, parent=self)
            else:
                tree = self.right_child.insert(*value)
            self.right_child = tree
        self.nodes.append(tree.nodes[-1])
        # if any child branch is too big, reconstruct the tree itself
        threshold = 0.8
        if (self.left_child is not None and float(self.left_child.size)/self.size > threshold) or (self.right_child is not None and float(self.right_child.size)/self.size > threshold):
            # print "Before reconstruct:"
            # self.printme()
            # print "After reconstruct:"
            new_tree = construct_tree(self.nodes, self.allnodes, self.parent)
            self.__dict__ = new_tree.__dict__
            # self.printme()
        return self


# construct a kd_tree from a list of nodes, return a tree with None parent. Note that the structure of nodes is changed
def construct_tree(nodes, allnodes=None, parent=None):
    if allnodes is None:    # allnodes is not None when re-construct a tree
        allnodes = nodes
    if nodes == []:
        return None
    if len(nodes) == 1:
        return Tree(nodes[0], allnodes, new_node=False, parent=parent)
    var_list = cal_variance([node.q for node in nodes])
    max_var = 0
    for i in range(len(var_list)):
        if var_list[i] >= max_var:
            max_var = var_list[i]
            split = i
    def get_sort_criterion(node):
        return node.q[split]
    nodes.sort(key=get_sort_criterion)
    tree = Tree(nodes[len(nodes)/2], allnodes, new_node=False, nodes=nodes, split=split, parent=parent)
    left_nodes = nodes[:len(nodes)/2]
    right_nodes = nodes[len(nodes)/2+1:]
    tree.left_child = construct_tree(left_nodes, allnodes, tree)
    tree.right_child = construct_tree(right_nodes, allnodes, tree)
    return tree

    # # return the last nodes connected to the tree
    # def extend(self, q_rand, env, robot, handles):
    #     # find the nearest node
    #     min_distance = inf
    #     node_nearest = None
    #     for node in self.nodes:
    #         distance_tmp = compute_distance(node.q, q_rand)
    #         if distance_tmp < min_distance:
    #             min_distance = distance_tmp
    #             node_nearest = node
    #
    #     # connect the nearest node to the sampled node q_rand
    #     node_new = node_nearest.copy()
    #     q_new = node_new.q
    #     distance_tmp = compute_distance(q_new, q_rand)
    #     while distance_tmp > self.dq:
    #         delta_q = q_rand - q_new
    #         if delta_q[4] > pi:
    #             delta_q[4] = delta_q[4] - 2*pi
    #         if delta_q[4] < -pi:
    #             delta_q[4] = delta_q[4] + 2*pi
    #         q_new = q_new + self.dq * delta_q / linalg.norm(delta_q)        # take a step forward
    #         if q_new[4] > pi:
    #             q_new[4] = q_new[4] - 2*pi
    #         if q_new[4] < -pi:
    #             q_new[4] = q_new[4] + 2*pi
    #         robot.SetActiveDOFValues(q_new)
    #         if env.CheckCollision(robot):
    #             return node_new
    #         else:
    #             # EE_position_new = GetEETransform(robot, q_new)[0:3, 3]
    #             # EE_position_old = GetEETransform(robot, node_new.q)[0:3, 3]
    #             # handles.append(env.drawlinestrip(points=array([EE_position_new, EE_position_old]),
    #             #                                  linewidth=0.02,
    #             #                                  colors=array([[1, 1, 0], [1, 1, 0]])))
    #             node_new = self.add_leaf(q_new, node_new)
    #             distance_tmp = compute_distance(q_new, q_rand)
    #     # EE_position_new = GetEETransform(robot, q_rand)[0:3, 3]
    #     # EE_position_old = GetEETransform(robot, node_new.q)[0:3, 3]
    #     # handles.append(env.drawlinestrip(points=array([EE_position_new, EE_position_old]),
    #     #                                  linewidth=0.02,
    #     #                                  colors=array([[1, 1, 0], [1, 1, 0]])))
    #     return self.add_leaf(q_rand, node_new)

# used for test
if __name__ == '__main__':
    NA = Node(0, np.array([0, -0.2, 0]), np.array([1, 1, 1]), 0)
    TA = Tree(NA, [])
    TA.insert(np.array([1, -0.1, 0]), np.array([2, 2, 2]), 0)
    TA.insert(np.array([-1, 0, 0]), np.array([2, 2, 2]), 0)
    TA.insert(np.array([-2, 10, 0]), np.array([2, 2, 2]), 0)
    TA.insert(np.array([-3, -10, 0]), np.array([2, 2, 2]), 0)
    node, dist = TA.find_nearest_node(*np.array([0, 0.1, 0]))
    # TA.printme()
    # T = construct_tree(TA.allnodes)
    # print "-----------------------------------------------------------------"
    # T.printme()
