import numpy as np

class Compute_Dist():
    
    def __init__(self, root):
        self.root = root

    #checks that L is only 0 on the diagonal, and 
    # the max distance is between the first and last items
    # perhaps all horizontal neighbors less than num_discrete_steps apart in L
    # should be different, assuming only hparams are floats (not switches)
    def check_distances_correct(self, L):
        max_L = float('-inf')
        max_L_index = []
        min_L = float('inf')
        min_L_index = []
        for i in range(len(L)):
            for j in range(len(L)):
                if L[i, j] > max_L:
                    max_L = L[i, j]
                    max_L_index = [[i, j]]
                elif L[i, j] == max_L:
                    max_L_index.append([i, j])
                if L[i, j] < min_L:
                    min_L = L[i, j]
                    min_L_index = [[i, j]]
                elif L[i, j] == min_L:
                    min_L_index.append([i, j])
        print("min_L: ", min_L)
        print("min_L_index: ", min_L_index)
        print("max_L: ", max_L)
        print("max_L_index: ", max_L_index)
        

    def compute_distance(self, first, second):
        return self.compute_pair_distance(first, second, self.root)


    #if first is none and second isn't, switch them. 
    #this way we only have to check if second is none
    def compute_pair_distance(self, first, second, node):
        if first is None and second is None:
            return 0
        elif first is None and second is not None:
            return self.compute_pair_distance(second, first, node)
        elif node.name == 'dict':
            return self.dict_distance(first, second, node)
        elif node.name == 'switch':
            return self.switch_distance(first, second, node)
        elif node.name == 'pos_args':
            return self.pos_args_distance(first, second, node)
        else:
            return self.leaf_distance(first, second, node)

    def dict_distance(self, first, second, node):
        cur_dist = 0
        for name, child in node.named_args:
            if second is not None:
                cur_dist += self.compute_pair_distance(first[name], second[name], child)
            else:
                cur_dist += self.compute_pair_distance(first[name], second, child)
        return cur_dist

    def switch_distance(self, first, second, node):
        cur_dist = 0
        for i in range(len(node.pos_args)-1):
            if second is not None:
                cur_dist += self.compute_pair_distance(first[i], second[i], node.pos_args[i+1])
            else:
                cur_dist += self.compute_pair_distance(first[i], second, node.pos_args[i+1])
        return cur_dist
        
    def pos_args_distance(self, first, second, node):
        cur_dist = 0
        for i in range(len(node.pos_args)):
            if second is not None:
                cur_dist += self.compute_pair_distance(first[i], second[i], node.pos_args[i])
            else:
                cur_dist += self.compute_pair_distance(first[i], second, node.pos_args[i])
        return cur_dist

    def leaf_distance(self, first, second, node):
        if second is None:
            return 1
        elif first == second:
            return 0
        elif node.name == 'literal':
            return 1
        elif node.name == 'float':
            distribution = node.pos_args[0].pos_args[1].name
            handler = getattr(self, '%s_distance' % distribution)
            return handler(first, second, node)
    
    def uniform_distance(self, first, second, node):
        lower_bound = node.pos_args[0].pos_args[1].pos_args[0].obj
        upper_bound = node.pos_args[0].pos_args[1].pos_args[1].obj
        return abs(1.0*max(first, second) - min(first, second))/abs(upper_bound - lower_bound)
        
    def loguniform_distance(self, first, second, node):
        return self.uniform_distance(np.log(first), np.log(second), node)
    
    def quniform_distance(self, first, second, node):
        return self.uniform_distance(first, second, node)
            
        
