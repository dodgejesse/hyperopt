"""
THIS IS INCOMPLETE!

the goal here was to draw hparam settings so that shorter trees had higher probability of being drawn.
we now think the best way is to sample uniformly from the set of possible trees

not sure how to do that yet.

"""


import numpy as np
import sys

def draw_one_sample(expr):
    hparam = {}
    vector = []
    make_vect(hparam, vector, expr, True)
    vector.append(1)
    return hparam, vector


def make_vect(hparams, vect, node, enter_vals):
    if node.name == 'dict':
        dict_vect(hparams, vect, node, enter_vals)
    elif node.name == 'switch':
        switch_vect(hparams, vect, node, enter_vals)
    elif node.name == 'pos_args':
        pos_args_vect(hparams, vect, node, enter_vals)
    elif node.name == 'float':
        float_vect(hparams, vect, node, enter_vals)
    elif node.name == 'literal':
        return
    else:
        raise ValueError("some kind of leaf node that isn't supported when " + 
                         "making discrete hparams vectors!")

def dict_vect(hparams, vect, node, enter_vals):
    for name, child in node.named_args:
        make_vect(hparams, vect, child, enter_vals)

def switch_vect(hparams, vect, node, enter_vals):
    switch_name = node.pos_args[0].pos_args[0].obj
    length_of_switch = len(node.pos_args)-1
    
    # case where we are below a turned-off switch
    if not enter_vals:
        hparams[switch_name] = []
        vect.append(0)
        for i in range(length_of_switch):
            vect.append(0)
        # this loop was pulled out to keep the order of vect more meaningful
        for i in range(length_of_switch):
            make_vect(hparams, vect, node.pos_args[i+1], False)
    # case where we are not below turned-off switch
    else:
        active_switch = np.random.choice(range(length_of_switch))
        hparams[switch_name] = [active_switch]
            
        for i in range(length_of_switch):
            if not i == active_switch:
                vect.append(0)
            else:
                vect.append(1)
        # this loop was pulled out to keep the order of vect more meaningful
        for i in range(length_of_switch):
            make_vect(hparams, vect, node.pos_args[i+1], active_switch == i)

def pos_args_vect(hparams, vect, node, enter_vals):
    for i in range(len(node.pos_args)):
        make_vect(hparams, vect, node.pos_args[i], enter_vals)

def float_vect(hparams, vect, node, enter_vals):
    float_name = node.pos_args[0].pos_args[0].obj
    lower_bound = node.pos_args[0].pos_args[1].pos_args[0].obj
    upper_bound = node.pos_args[0].pos_args[1].pos_args[1].obj
    if not enter_vals:
        hparams[float_name] = []
        vect.append(0)
        return
    elif lower_bound == upper_bound:
        hparams[float_name] = [lower_bound]
        return
    else:
        distribution = node.pos_args[0].pos_args[1].name
        cur_module = sys.modules[__name__]
        handler = getattr(cur_module, '%s_distance' % distribution)
        unif_sample = random.uniform(lower_bound, upper_bound)
        vect.append(unif_sample)
        handler(hparams, unif_sample, node, float_name)

def uniform_distance(hparam, unif_sample, node, float_name):
    hparam[float_name] = [unif_sample]
    #vect.append((1.0*hparam - lower_bound)/(upper_bound - lower_bound))

def loguniform_distance(hparam, unif_sample, node, float_name):
    # have to scale correctly here
    uniform_distance(np.log(hparam), vect, node, float_name)

def quniform_distance(hparam, unif_sample, node, float_name):
    # have to mulitply it into the right range here
    uniform_distance(hparam, vect, node, float_name)
