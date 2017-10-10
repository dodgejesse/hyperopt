"""
THIS IS INCOMPLETE!

the goal here was to draw hparam settings so that shorter trees had higher probability of being drawn.
we now think the best way is to sample uniformly from the set of possible trees

not sure how to do that yet.

"""


import numpy as np
import sys
import hparam_distribution_sampler
import copy

# state: contains the tree-structured-space
# actions: can draw one sample uniformly from that space
class Unif_Sampler():
    def __init__(self, space):
        self.space = space
        self.dists = []
        self.index_names = []
        self.switch_index_to_children = {}
        self.switch_index_to_length = {}
        self.hparam_out = {}
        self.process_node_dists(space, [])


    # draws a uniform sample which is a valid tree
    def draw_unif_samp(self):
        valid_samp = False
        sample = []
        counter = 0
        while not valid_samp:
            sample = self.sample_indepently()
            valid_samp = self.evaluate_sample(sample)
            counter += 1
        #print ("DEBUGGING: took {} tries to get valid sample".format(counter))
        self.kill_children(sample)
        return sample

    # turns off floats that are under off switches
    def kill_children(self, sample):
        for parent in self.switch_index_to_children:
            for child in self.switch_index_to_children[parent]:
                if not sample[parent]:
                    
                    # unit test to test that if the parent is off, the child is not (on and a switch)
                    child_is_switch = child in self.switch_index_to_children
                    child_is_on = sample[child] == 1
                    assert not (child_is_switch and child_is_on)


                    sample[child] = []



    # helper method for draw_unif_samp
    # draws a sample which might not be a valid tree. each element sampled independently
    def sample_indepently(self):
        indep_samps = []
        for i in range(len(self.dists)):
            indep_samps.append(self.dists[i].draw_sample())
        return indep_samps

    # helper method for draw_unif_samp
    # checks to see if a given independent sample is a valid tree
    def evaluate_sample(self, sample):
        # checks if the children of an off switch are off
        # only checks other switches
        for parent in self.switch_index_to_children:
            for i in range(len(self.switch_index_to_children[parent])):
                child_index = self.switch_index_to_children[parent][i]
                # check that the child is also a switch
                if child_index in self.switch_index_to_children:
                    if sample[child_index] and not sample[parent]:
                        return False

        # check if for each switch, exactly one is on
        for first_switch_sibling in self.switch_index_to_length:
            num_on = 0
            for i in range(self.switch_index_to_length[first_switch_sibling]):
                num_on += sample[first_switch_sibling + i]
            if not num_on == 1:
                return False

        # passed our tests
        return True



    # takes a uniformly drawn sample and scales and shifts each element to [0,1].
    def make_zero_one_vect(self, sample):
        zero_one_vect = []
        #import pdb; pdb.set_trace()
        for i in range(len(sample)):
            zero_one_vect.append(self.dists[i].make_zero_one(sample[i]))
            #print i, sample[i], self.dists[i].make_zero_one(sample[i]), self.index_names[i]
    
        return zero_one_vect
        #print "DEBUGGING"
        #for i in range(len(sample)):
        #    print self.index_names[i], sample[i], zero_one_vect[i]


    # takes a sample and returns a valid hparam set to return
    def make_full_hparam_list(self, sample):
        hparams_to_return = copy.deepcopy(self.hparam_out)
        self.add_switch_values(sample, hparams_to_return)
        self.add_float_values(sample, hparams_to_return)
        return hparams_to_return

    # helper method for make_full_hparam_list
    # adds values for switches from the sample to the output list
    def add_switch_values(self, sample, hparams_to_return):
        
        for switch_index in self.switch_index_to_length:
            switch_name = self.index_names[switch_index][:-2]

            # if the switch is below a switch that's off
            if sample[switch_index] == []:
                hparams_to_return[switch_name] = []
                
            counter = 0
            for i in range(self.switch_index_to_length[switch_index]):
                if sample[switch_index + i] == 1:
                    hparams_to_return[switch_name] = [counter]
                    break
                counter += 1
            

    # helper method for make_full_hparam_list
    # adds values for floats from the sample to the output list
    def add_float_values(self, sample, hparams_to_return):
        # a list of the indices that point to switches
        switch_indices = []
        for first_switch_index in self.switch_index_to_length:
            for i in range(self.switch_index_to_length[first_switch_index]):
                switch_indices.append(first_switch_index + i)


        for i in range(len(sample)):
            if i not in switch_indices:
                if sample[i] == []:
                    hparams_to_return[self.index_names[i]] = sample[i]
                else:
                    hparams_to_return[self.index_names[i]] = [sample[i]]
                    


            
    def add_dist_and_name(self, sampler, name, switch_parents):
        cur_index = len(self.dists)
        self.dists.append(sampler)
        self.index_names.append(name)
        for switch_parent in switch_parents:
            if not switch_parent in self.switch_index_to_children:
                self.switch_index_to_children[switch_parent] = []
            self.switch_index_to_children[switch_parent].append(cur_index)
            

    def process_node_dists(self, node, switch_parents):
        if node.name == 'dict':
            self.dict_vect(node, switch_parents)
        elif node.name == 'switch':
            self.switch_vect(node, switch_parents)
        elif node.name == 'pos_args':
            self.pos_args_vect(node, switch_parents)
        elif node.name == 'float':
            self.float_vect(node, switch_parents)
        elif node.name == 'literal':
            return
        else:
            raise ValueError("some kind of leaf node that isn't supported when " + 
                             "making discrete hparams vectors!")

    def dict_vect(self, node, switch_parents):
        for name, child in node.named_args:
            self.process_node_dists(child, switch_parents)

    def switch_vect(self, node, switch_parents):
        switch_name = node.pos_args[0].pos_args[0].obj
        self.hparam_out[switch_name] = [0]
        length_of_switch = len(node.pos_args)-1
        if length_of_switch > 1:
            switch_index = len(self.dists)
            self.switch_index_to_length[switch_index] = length_of_switch
            for i in range(length_of_switch):
                self.add_dist_and_name(hparam_distribution_sampler.Bernoulli(.5),
                                       "{}_{}".format(switch_name, i), switch_parents)

        for i in range(length_of_switch):
            # add this node as parent in switch_parents, recurse, then remove
            if length_of_switch > 1:
                switch_parents.append(switch_index + i)
            self.process_node_dists(node.pos_args[i+1], switch_parents)
            if length_of_switch > 1:
                del switch_parents[-1]




    def pos_args_vect(self, node, switch_parents):
        for i in range(len(node.pos_args)):
            self.process_node_dists(node.pos_args[i], switch_parents)

    def float_vect(self, node, switch_parents):
        float_name = node.pos_args[0].pos_args[0].obj
        lower_bound = node.pos_args[0].pos_args[1].pos_args[0].obj
        upper_bound = node.pos_args[0].pos_args[1].pos_args[1].obj
        

        sampler = self.get_sampler(node, lower_bound, upper_bound)
        if node.pos_args[0].pos_args[1].name == 'loguniform':
            self.hparam_out[float_name] = [np.exp(lower_bound)]
        else:
            self.hparam_out[float_name] = [lower_bound]

        if lower_bound == upper_bound:
            return

        self.add_dist_and_name(sampler, float_name, switch_parents)
        


    def get_sampler(self, node, lower_bound, upper_bound):
        distribution = node.pos_args[0].pos_args[1].name
        if distribution == 'uniform':
            return hparam_distribution_sampler.Uniform(lower_bound, upper_bound)
        elif distribution == 'loguniform':
            return hparam_distribution_sampler.LogUniform(lower_bound, upper_bound)
        elif distribution == 'quniform':
            return hparam_distribution_sampler.QUniform(lower_bound, upper_bound, 
                                                        node.pos_args[0].pos_args[1].pos_args[2].obj)
        else:
            print("We don't have a distribution implemented for {}".format(distribution))
            raise NameError("We don't have a distribution implemented for {}".format(distribution))





