import numpy as np

class Make_Vector():
    
    def __init__(self, root):
        self.root = root

    def make_vectors(self, hparam_sets):
        vectors = []
        for hparam_set in hparam_sets:
            vect = []
            self.make_vect(hparam_set, vect, self.root)
            vect_as_array = np.asarray(vect)
            vectors.append(vect_as_array/np.linalg.norm(vect_as_array,2))
        
        return vectors


    def make_vect(self, hparams, vect, node):
        if node.name == 'dict':
            self.dict_vect(hparams, vect, node)
        elif node.name == 'switch':
            self.switch_vect(hparams, vect, node)
        elif node.name == 'pos_args':
            self.pos_args_vect(hparams, vect, node)
        else:
            self.leaf_vect(hparams, vect, node)

    def dict_vect(self, hparams, vect, node):
        for name, child in node.named_args:
            #if hparams is None, we're in an "off" switch branch
            if hparams is not None:
                self.make_vect(hparams[name], vect, child)
            else:
                self.make_vect(hparams, vect, child)

    def switch_vect(self, hparams, vect, node):
        for i in range(len(node.pos_args)-1):
            if hparams is not None:
                self.make_vect(hparams[i], vect, node.pos_args[i+1])
            else:
                self.make_vect(hparams, vect, node.pos_args[i+1])
    
    def pos_args_vect(self, hparams, vect, node):
        for i in range(len(node.pos_args)):
            if hparams is not None:
                self.make_vect(hparams[i], vect, node.pos_args[i])
            else:
                self.make_vect(hparams, vect, node.pos_args[i])

    def leaf_vect(self, hparam, vect, node):
        if hparam is None:
            vect.append(0)
        elif node.name == 'literal':
            vect.append(1)
        elif node.name == 'float':
            distribution = node.pos_args[0].pos_args[1].name
            handler = getattr(self, '%s_distance' % distribution)
            vect.append(handler(hparam, node))
    
    def uniform_distance(self, hparam, node):
        lower_bound = node.pos_args[0].pos_args[1].pos_args[0].obj
        upper_bound = node.pos_args[0].pos_args[1].pos_args[1].obj
        return  (1.0*hparam - lower_bound)/(upper_bound - lower_bound)
        
    def loguniform_distance(self, hparam, node):
        return self.uniform_distance(np.log(hparam), node)
    
    def quniform_distance(self, hparam, node):
        return self.uniform_distance(hparam, node)
            
        
