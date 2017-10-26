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
#import dpp_sample_compiled_matlab
import time
import sys
import scipy.spatial
import dpp_mcmc_sampler

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



def check_sampled_points_more_diverse(L, distance, vectors, d_space, k):
    dist_map = {"cos":"cosine", "l2":"euclidean", "ham":"hamming"}
    set_size = k
    dpp_avg = 0
    mcmc_avg = 0
    rand_avg = 0
    num_sets = 2500

    exact = False
    if exact:
        # DEBUG to make it go faster
        import DPP_Sampler
        import matlab
        dpp_samp = DPP_Sampler.initialize()
        L_list = np.ndarray.tolist(L)
        L_mat = matlab.double(L_list)
        L_decomp = dpp_samp.decompose_kernel(L_mat)

    for i in range(num_sets):

        #import pdb; pdb.set_trace()
        
        mcmc_sampled_indices = dpp_mcmc_sampler.sample_k(range(len(L)), L, set_size)
        cur_mcmc_avg = np.average(scipy.spatial.distance.pdist(vectors[mcmc_sampled_indices],dist_map[distance]))
        mcmc_avg = (mcmc_avg * i + cur_mcmc_avg)/(i+1)
        

        rand_sampled_indices = random.sample(range(len(d_space)), set_size)
        cur_rand_avg = np.average(scipy.spatial.distance.pdist(vectors[rand_sampled_indices],dist_map[distance]))
        rand_avg = (rand_avg * i + cur_rand_avg)/(i+1)

        if exact:
            dpp_sampled_indices_matlab = dpp_samp.sample_dpp(L_decomp,random.randint(1,999999),set_size)
            dpp_sampled_indices = [int(index[0])-1 for index in dpp_sampled_indices_matlab]
            cur_dpp_avg = np.average(scipy.spatial.distance.pdist(vectors[dpp_sampled_indices],dist_map[distance]))
            dpp_avg = (dpp_avg * i + cur_dpp_avg)/(i+1)



        print('iter {}: {}, {}, {}'.format(i,rand_avg, mcmc_avg, dpp_avg))
    if exact:
        dpp_samp.terminate()        
    print('rand_avg: {}'.format(rand_avg))
    print('mcmc_avg: {}'.format(mcmc_avg))
    print('dpp_avg: {}'.format(dpp_avg))

    


# to linearly rescale matrix X from [min,max] to [a,b]:
# f(x) = ((b-a)(x-min))/(max-min)+a
#      = (x) (b-a)/(max-min)-(b-a)(min)/(max-min)+a
def scale_and_shift(mi, ma, a, b, X):
    
    mult = 1.0*(b-a)/(ma-mi)
    add = -(b-a)*(mi)*1.0/(ma-mi)+a
    X_scaled = np.multiply(X, mult)
    X_prime = X_scaled + add
    return X_prime


def generate_L_from_vectors(vectors, distance):
    #import pdb; pdb.set_trace()
    if distance == 'rbf':
        import sklearn.metrics
        return sklearn.metrics.pairwise.rbf_kernel(vectors, vectors)
    dist_map = {"cos":"cosine", "l2":"euclidean", "ham":"hamming"}
    debug_dists = False
    if debug_dists:
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


def sample_discrete_dpp(trials, domain, seed):
    #import pdb; pdb.set_trace() 
    discretizer = Discretizer(trials.discretize_num)
    d_space = discretizer.discretize_space(domain)

    make_vect = Make_Vector(domain.expr)

    distance = trials.dpp_dist
    vectors = np.asarray(make_vect.make_vectors(d_space, distance))

    L = generate_L_from_vectors(vectors, distance)

    check_diversity = False
    if check_diversity:
        check_sampled_points_more_diverse(L, distance, vectors, d_space, trials.max_evals)
        

    start_sample_time = time.time()
    dpp_sampled_indices = dpp_mcmc_sampler.sample_k(range(len(L)), L, trials.max_evals)
    #dpp_sampled_indices = dpp_sample_compiled_matlab.sample_dpp(L, seed, trials.max_evals)
    print("sampling {} items from a DPP of size {} took {} seconds".format(trials.max_evals, 
                    len(L), time.time() - start_sample_time))

    return [d_space[int(index)] for index in dpp_sampled_indices]

# call the mcmc algorithm        
def sample_continuous_dpp(trials, domain, seed):
    #import pdb; pdb.set_trace()
    if trials.dpp_dist == 'rbf':
        import rbf_kernel
        dist_fn = rbf_kernel.RBF_Kernel()
    elif trials.dpp_dist == 'rbf_clip':
        import rbf_kernel
        dist_fn = rbf_kernel.RBF_Clipped_Kernel('k')
    elif trials.dpp_dist == 'rbf_narrow':
        import rbf_kernel
        #DEBUGGING
        # this bandwidth is just for ICLR
        bandwidth=8
        dist_fn = rbf_kernel.RBF_Kernel(bandwidth)

    # for some reason this has to be imported here...?
    from unif_hparam_sample import Unif_Sampler
    unif_sampler = Unif_Sampler(domain.expr)

    k = trials.max_evals
    d = len(unif_sampler(1)[1])
    num_iters = int(max(1000, np.power(k,3) * d))
    
    unfeat_B_Y, B_Y, L_Y, time = dpp_mcmc_sampler.sample_k_disc_and_cont(unif_sampler, dist_fn, k, max_iter=num_iters)

    hparam_assignments = unif_sampler.make_full_hparam_list_set(unfeat_B_Y)


    #DEBUG
    #num_samples_to_draw = 20
    #for i in range(num_samples_to_draw):
    #    unif_samp = unif_sampler.draw_unif_samp()
    #    zero_one_vect = unif_sampler.make_zero_one_vect(unif_samp)
    #    print zero_one_vect
        #print unif_sampler.make_full_hparam_list(unif_samp)
    #print('')

    

    return hparam_assignments
    

def suggest(new_ids, domain, trials, seed, *args, **kwargs):
    

    #if first time through, sample set of hparams
    if new_ids[0] == 0:
        if trials.discretize_space:
            hparams_to_try = sample_discrete_dpp(trials, domain, seed)
        else:
            hparams_to_try = sample_continuous_dpp(trials, domain, seed)

        
        print("The hyperparameter settings that will be evaluated " + 
              "(evaluation order will be uniformly sampled):")
        for thing in hparams_to_try:
            print thing
        print("")

        random.shuffle(hparams_to_try)
        trials.hparams_to_try = []
        for i in range(len(hparams_to_try)):
            trials.hparams_to_try.append(output_format(hparams_to_try[i], i, domain, trials))

        
    return trials.hparams_to_try[new_ids[0]]


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
        
    
    
