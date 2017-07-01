"""
Annealing algorithm for hyperopt

Annealing is a simple but effective variant on random search that
takes some advantage of a smooth response surface.

The simple (but not overly simple) code of simulated annealing makes this file
a good starting point for implementing new search algorithms.

"""

__authors__ = "mostly Jesse Dodge, but influenced by James Bergstra"
__license__ = "3-clause BSD License"
__contact__ = "github.com/jaberg/hyperopt"


import numpy as np
from hparam_as_vector import Make_Vector
import random
import copy
from discretize_space import Discretizer
from discretized_distance import Compute_Dist
import dpp_sample_compiled_matlab
import time
import sys
import scipy.spatial


def get_num_quantiles(node):
    if node.name == 'literal':
        return 1
    elif node.name == 'pos_args':
        n_q = 1
        for element in node.pos_args:
            n_q = n_q * get_num_quantiles(element)
        return n_q
    elif node.name == 'float':
         dist = node.pos_args[0].pos_args[1].name



def avg_dist_of_set(sampled_items, distance_calc, max_L, min_L):
    avg_dist = 0
    counter = 0
    for j in range(len(sampled_items)):
        for k in range(j,len(sampled_items)):
            if j == k:
                continue
            cur_dist = distance_calc(sampled_items[j], sampled_items[k])
            counter += 1
            if max_L is not None and min_L is not None:
                avg_dist += 2*(cur_dist-min_L)/(max_L-min_L)-1
            else:
                avg_dist += cur_dist
            
    return avg_dist / counter



def check_sampled_points_more_diverse(L,max_L,min_L, distance_calc, d_space,k):
    set_size = k
    dpp_avg = 0
    rand_avg = 0
    num_sets = 2500
    for i in range(num_sets):
        #this next line is how we call the mcmc algorithm
        #dpp_sampled_items = dpp_sampler.sample_k(d_space, L, set_size, max_nb_iterations = 10000)
        dpp_sampled_indices = dpp_sampler.dpp.sample_dpp(L, k)
        dpp_sampled_items = [d_space[index] for index in dpp_sampled_indices]
        cur_dpp_avg = avg_dist_of_set(dpp_sampled_items, distance_calc, max_L, min_L)
        
        dpp_avg = (dpp_avg * i + cur_dpp_avg)/(i+1)
        
        
        rand_sampled_items = random.sample(d_space, set_size)
        cur_rand_avg = avg_dist_of_set(rand_sampled_items, distance_calc, max_L, min_L)
        rand_avg = (rand_avg * i + cur_rand_avg)/(i+1)
        print('iter {}: {}, {}'.format(i,rand_avg, dpp_avg))
        
    print('dpp_avg: {}'.format(dpp_avg))
    print('rand_avg: {}'.format(rand_avg))


# to linearly rescale matrix X from [min,max] to [a,b]:
# f(x) = ((b-a)(x-min))/(max-min)+a
#      = (x) (b-a)/(max-min)-(b-a)(min)/(max-min)+a
def scale_and_shift(mi, ma, a, b, X):
    mult = (b-a)/(ma-mi)
    add = -(b-a)*(mi)/(ma-mi)+a
    X_scaled = np.multiply(X, mult)
    X_prime = X_scaled + add
    return X_prime


def generate_L_from_vectors(vectors, distance):
    dist_map = {"cos":"cosine", "l2":"euclidean", "ham":"hamming"}
    for dist in dist_map:
        dists = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(vectors,dist_map[dist]))
        if dist != 'l2':
            unscaled = 1-dists
            print("unscaled {} matrix rank: {}".format(dist, np.linalg.matrix_rank(unscaled)))
        L = 1 - scale_and_shift(np.min(dists), np.max(dists), 0, 1, dists)
        print("scaled_and_shifted(0,1) {} matrix rank: {}".format(dist,np.linalg.matrix_rank(L)))
        L = -1 * scale_and_shift(np.min(dists), np.max(dists), -1, 1, dists)
        print("scaled_and_shifted(-1,1) {} matrix rank: {}".format(dist,np.linalg.matrix_rank(L)))
            

    dists = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(vectors,dist_map[distance]))
    L = 1 - scale_and_shift(np.min(dists), np.max(dists), 0, 1, dists)
    
    # to make it symmetric and to make it more likely to be psd
    #L_sym = (np.transpose(L)+L)*(1.0/2)
    #L_prime = L_sym + np.identity(len(L_sym))*(np.power(10.0,-14))


    return L
    


#DEBUGGING
#This is super janky. i don't know the correct way to make the output format, so i'm guessing
#should replicate what hyperopt.pyll.base.rec_eval does to make the 'vals' object, 
#but i can't figure out how it does it.  
# also mostly coping algobase.SuggestAlgo.__call__
def output_format(vals, new_id, domain, trials):
    idxs = {}
    for k in vals:
        if vals[k] == []:
            idxs[k] = []
        else:
            idxs[k] = [new_id]
    new_result = domain.new_result()

    new_misc = dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)
    #DEBUG not sure what this next line does
    from base import miscs_update_idxs_vals
    miscs_update_idxs_vals([new_misc], idxs, vals)
    rval = trials.new_trial_docs([new_id], [None], [new_result], [new_misc])
    return rval

def construct_L_debug(num_points):
    step_size = 1.0/(num_points-1)
    L_debug = []
    for i in range(num_points):
        cur_row = []
        for j in range(num_points):
            cur_row.append(1-abs(step_size*(i-j)))
        L_debug.append(cur_row)
        
    items = []
    for i in range(num_points):
        items.append(i)
    return items, np.asarray(L_debug)
    

def debug_mcmc(d_space, L):
    np.set_printoptions(linewidth=20000)
    import dpp_mcmc_sampler

    items, L_debug = construct_L_debug(10)
    
    #things = dpp_mcmc_sampler.sample_k(items, L_debug, 3)


    items, L_debug = construct_L_debug(4080)
    #dpp_mcmc_sampler.sample_k(items, L, 3)
    #dpp_mcmc_sampler.sample_k(items, L, 4)


    
    dpp_mcmc_sampler.sample_k(items, L, 7)



    import pdb; pdb.set_trace()
    #L_small = np.array([[1,.6,.3],[.6,1,.6],[.3,.6,1]])
    #items = ['1','2','3']

    #things = dpp_mcmc_sampler.sample_k(items, L_small, 2)
    
    #L_s = np.array([[1,.9,.8,.7],[.9,1,.9,.8],[.8,.9,1,.9],[.7,.8,.9,1]])
    #items = ['1','2','3','4']

    
    
    #things = dpp_mcmc_sampler.sample_k(items, L_s, 2)


    #things = dpp_mcmc_sampler.sample_k(d_space, L, 5)


def suggest(new_ids, domain, trials, seed, *args, **kwargs):
    import pdb; pdb.set_trace()

    #if first time through, sample set of hparams
    if new_ids[0] == 0:
        discretizer = Discretizer(trials.discretize_num)
        d_space = discretizer.discretize_space(domain)

        make_vect = Make_Vector(domain.expr)


        distance = trials.dpp_dist
        vectors = np.asarray(make_vect.make_vectors(d_space, distance))

        L = generate_L_from_vectors(vectors, distance)


        debug_mcmc(d_space, L)





        
        check_diversity = False
        if check_diversity:
            distance_calc = Compute_Dist(domain.expr)
            check_sampled_points_more_diverse(L, None, None, distance_calc.compute_distance, d_space, 5)
        
        print_dpp_samples = False
        if print_dpp_samples:
            print("ABOUT TO START PRINTING DPP SAMPLES")
            print("")
            for i in range(100):
                dpp_sampled_indices = dpp_sample_compiled_matlab.sample_dpp(L, np.random.randint(seed), trials.max_evals)
                points = []
                for index in dpp_sampled_indices:
                    points.append(d_space[int(index[0])-1])
                    points[len(points)-1]['index'] = int(index[0]-1)
                
                for thing in points:
                    print thing
                print("")
            sys.exit(0)

        start_sample_time = time.time()
        dpp_sampled_indices = dpp_sample_compiled_matlab.sample_dpp(L, seed, trials.max_evals)
        print("sampling {} items from a DPP of size {} took {} seconds".format(trials.max_evals, 
                            len(L), time.time() - start_sample_time))

        
        trials.dpp_sampled_points = [d_space[int(index[0])-1] for index in dpp_sampled_indices]
        print("The hyperparameter settings that will be evaluated:")
        for thing in trials.dpp_sampled_points:
            print thing
        print("")
        random.shuffle(trials.dpp_sampled_points)
    return output_format(trials.dpp_sampled_points[new_ids[0]], new_ids[0], domain, trials)


def time_dpp(domain, trials, num_discrete_steps=11, k=None, print_L_time=False):

    discretizer = Discretizer(num_discrete_steps)
    d_space = discretizer.discretize_space(domain)


    start_vect_time = time.time()
    make_vect = Make_Vector(domain.expr)
    vectors = np.asarray(make_vect.make_vectors(d_space))
    if print_L_time:
        print("took {} seconds to make vectors (B)".format(time.time() - start_vect_time))
    start_L_time = time.time()
    L = generate_L_from_vectors(vectors)
    if print_L_time:
        print("took {} seconds to make L=B^TB".format(time.time() - start_L_time))
    
    if k == None:
        k = trials.max_evals
    dpp_sampled_indices = dpp_sampler.dpp.sample_dpp(L, k)
        
    
    
