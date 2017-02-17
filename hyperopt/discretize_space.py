import numpy as np
import copy
from none_storage import None_storage
class Discretizer():
    num_discrete_steps = 10.0

    def increment_uniform(self, node, hp_out, step_size=None):
        lower_bound = node.pos_args[0].pos_args[1].pos_args[0].obj
        upper_bound = node.pos_args[0].pos_args[1].pos_args[1].obj
        if hp_out is None:
            return False, lower_bound
        if step_size is None:
            step_size = (upper_bound - lower_bound) / self.num_discrete_steps
        assert lower_bound <= hp_out <= upper_bound
        if hp_out + step_size >= upper_bound:
            return False, lower_bound
        else:
            return True, hp_out + step_size
            
    def increment_quniform(self,node, hp_out):
        return self.increment_uniform(node, hp_out, 
                                      step_size=node.pos_args[0].pos_args[1].pos_args[2].obj)

    def increment_loguniform(self,node, hp_out):
        
        if hp_out is not None:
            hp_out = np.log(hp_out)
        to_return = self.increment_uniform(node, hp_out)
        return to_return[0], np.exp(to_return[1])

    #input:
    #node: the current hyperparam to be evaluated
    #hp_out: the current value for the hparam
    #
    #returns (true,new_value) if successfully incremented node
    #returns (false,new_value) if this is the first or only value it can take
    def increment_leaf(self, node, hp_out):
        if node.name == 'float':
            distribution = node.pos_args[0].pos_args[1].name
            handler = getattr(self, 'increment_%s' % distribution)
            return handler(node, hp_out)
        elif node.name == 'literal':
            if node.obj is None:
                return False, None_storage()
            else:
                return False, node.obj
        else: 
            raise ValueError("some kind of leaf node that isn't supported in discritization!")

    def increment_pos_args(self, node, hp_out):
        if hp_out is None:
            cur_hp_out = []
            for child in node.pos_args:
                cur_hp_out.append(self.increment_node(child, None)[1])
            return False, cur_hp_out
        for i in range(len(hp_out)):
            incremented, new_value = self.increment_node(node.pos_args[i], hp_out[i])
            hp_out[i] = new_value
            if incremented:
                return True, hp_out
        return False, hp_out

    def increment_dict(self, node, hp_out):
        #first time through, needs to be set
        if hp_out is None:
            cur_hp_out = {}
            for name, child in node.named_args:
                cur_hp_out[name] = self.increment_node(child, None)[1]
            return False, cur_hp_out
        for name, child in node.named_args:
            incremented, new_value = self.increment_node(child, hp_out[name])
            hp_out[name] = new_value
            #if incremented is true, this updates hp_out with new value
            #if it's false, hp_out[name] is reset to first value
            if incremented:
                return True, hp_out
        return False, hp_out

    def increment_switch(self, node, hp_out):
        #for the first time through, when it's not set
        if hp_out is None:
            length_of_switch = len(node.pos_args)-1
            cur_hp_out = [None]*length_of_switch
            cur_hp_out[0] = self.increment_node(node.pos_args[1], None)[1]
            return False, cur_hp_out
        for i in range(len(hp_out)):
            if hp_out[i] is not None:
                incremented, new_value = self.increment_node(node.pos_args[i+1], hp_out[i])
                #case where we successfully incremented child i
                if incremented:
                    hp_out[i] = new_value
                    return True, hp_out
                #did not successfully increment i, and it's not the last child
                elif not i == len(hp_out) - 1:
                    hp_out[i] = None
                    hp_out[i+1] = self.increment_node(node.pos_args[i+2], None)[1]
        #did not successfully increment any node, so it should be reset
        return self.increment_switch(node, None)

    def increment_node(self,node, hp_out):
        if node.name == 'dict':
            return self.increment_dict(node, hp_out)
        elif node.name == 'switch':
            return self.increment_switch(node, hp_out)
        elif node.name == 'pos_args':
            return self.increment_pos_args(node, hp_out)
        else:
            return self.increment_leaf(node, hp_out)

    def discretize_space(self, domain):
        #set of hparam settings
        hp_out_set = []
        #current hparam setting
        root = domain.expr
        
        incremented, new_values = self.increment_node(root,None)
        incremented = True
        hp_out_set.append(copy.deepcopy(new_values))
        while incremented:
            incremented, new_values = self.increment_node(root, new_values)
            hp_out_set.append(copy.deepcopy(new_values))
        import pdb; pdb.set_trace()



"""
    def hp_uniform(self, memo, node, label, tid, val,
                   log_scale=False,
                   pass_q=False,
                   uniform_like=uniform):
        
        Return a new value for a uniform hyperparameter.

        Parameters:
        -----------

        memo - (see on_node_hyperparameter)

        node - (see on_node_hyperparameter)

        label - (see on_node_hyperparameter)

        tid - trial-identifier of the model trial on which to base a new sample

        val - the value of this hyperparameter on the model trial

        Returns: a list with one value in it: the suggested value for this
        hyperparameter
        
        if log_scale:
            val = np.log(val)
        high = memo[node.arg['high']]
        low = memo[node.arg['low']]
        assert low <= val <= high
        width = (high - low) * self.shrinking(label)
        new_high = min(high, val + width / 2)
        if new_high == high:
            new_low = new_high - width
        else:
            new_low = max(low, val - width / 2)
            if new_low == low:
                new_high = new_low + width
        assert low <= new_low <= new_high <= high
        if pass_q:
            return uniform_like(
                low=new_low,
                high=new_high,
                rng=self.rng,
                q=memo[node.arg['q']],
                size=memo[node.arg['size']])
        else:
            return uniform_like(
                low=new_low,
                high=new_high,
                rng=self.rng,
                size=memo[node.arg['size']])

    def hp_quniform(self, *args, **kwargs)

    def hp_loguniform(self, *args, **kwargs)

    def hp_qloguniform(self, *args, **kwargs)

    def hp_randint(self, memo, node, label, tid, val)

    def hp_categorical(self, memo, node, label, tid, val)

    def hp_normal(self, memo, node, label, tid, val)
        
    def hp_lognormal(self, memo, node, label, tid, val)

    def hp_qlognormal(self, memo, node, label, tid, val)

    def hp_qnormal(self, memo, node, label, tid, val)



    def process_switch(self, node, i, hp_out, parent_final_node):
        if len(node.pos_args) == i:
            return
        final_node = parent_final_node and (len(node.pos_args)-1 == i)
        #not sure if useful, but can get the number of elements in the switch here
        if node.pos_args[i].name == 'dict':
            self.process_dict(node.pos_args[i], 0, hp_out, final_node)
        elif node.pos_args[i].name == 'switch':
            self.process_switch(node.pos_args[i], 0, hp_out,final_node)
        else:
            self.increment_node(node.pos_args[i], hp_out, final_node)
        self.process_switch(node, i+1, hp_out, parent_final_node)

    def process_dict(self, node, i, hp_out, parent_final_node):
        if len(node.named_args) == i:
            return
        final_node = parent_final_node and (len(node.named_args)-1 == i)
        if node.named_args[i][1].name == 'dict':
            self.process_dict(node.named_args[i][1], 0, hp_out, final_node)
            self.process_dict(node, i+1, hp_out, parent_final_node)
        elif node.named_args[i][1].name == 'switch':
            self.process_switch(node.named_args[i][1], 1, hp_out,final_node)
            self.process_dict(node, i+1, hp_out, parent_final_node)
        else:
            #num_quantiles = get_num_quantiles(node.named_args[i][1])
            hp_out[node.named_args[i][0]] = 
            while self.increment_node(node.named_args[i][1],hp_out, final_node):
                self.process_dict(node, i+1, hp_out, parent_final_node)

    
"""
