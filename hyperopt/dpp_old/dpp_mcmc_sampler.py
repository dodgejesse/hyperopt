"""
Taken from https://github.com/mehdidc/dpp!
Don't forget to cite
DEBUGGING
"""

import numpy as np
from itertools import product
import pyll.stochastic
"""
Determinantal point process sampling procedures based
on  (Fast Determinantal Point Process Sampling with
     Application to Clustering, Byungkon Kang, NIPS 2013)
"""


def sample(items, L, max_nb_iterations=1000, rng=np.random):
    """
    Sample a list of items from a DPP defined
    by the similarity matrix L. The algorithm
    is iterative and runs for max_nb_iterations.
    The algorithm used is from
    (Fast Determinantal Point Process Sampling with
    Application to Clustering, Byungkon Kang, NIPS 2013)
    """
    Y = rng.choice((True, False), size=len(items))
    L_Y = L[Y, :]
    L_Y = L_Y[:, Y]
    L_Y_inv = np.linalg.inv(L_Y)

    for i in range(max_nb_iterations):
        u = rng.randint(0, len(items))

        c_u = L[u:u+1, :]
        c_u = c_u[:, u:u+1]
        b_u = L[Y, :]
        b_u = b_u[:, u:u+1]
        if Y[u] == False:
            p_include_U = min(1, c_u - np.dot(np.dot(b_u.T, L_Y_inv), b_u))
            if p_include_U > 0:
                print "p_include_U: {}".format(p_include_U)

            if rng.uniform() <= p_include_U:
                d_u = (c_u - np.dot(np.dot(b_u.T, L_Y_inv), b_u))
                upleft = (L_Y_inv +
                          np.dot(np.dot(np.dot(L_Y_inv, b_u), b_u.T),
                                 L_Y_inv) / d_u)
                upright = -np.dot(L_Y_inv, b_u) / d_u
                downleft = -np.dot(b_u.T, L_Y_inv) / d_u
                downright = d_u
                L_Y_inv = np.bmat([[upleft, upright], [downleft, downright]])
                Y[u] = True
                L_Y = L[Y, :]
                L_Y = L_Y[:, Y]
        else:
            p_remove_U = min(1, 1./(c_u - np.dot(np.dot(b_u.T, L_Y_inv), b_u)))
            if p_remove_U > 0:
                print "p_remove_U: {}".format(p_remove_U)
            if rng.uniform() <= p_remove_U:
                l = L_Y_inv.shape[0] - 1
                D = L_Y_inv[0:l, :]
                D = D[:, 0:l]
                e = L_Y_inv[0:l, :]
                e = e[:, l:l+1]
                f = L_Y_inv[l:l+1, :]
                f = f[:, l:l+1]
                L_Y_inv = D - np.dot(e, e.T) / f
                Y[u] = False
                L_Y = L[Y, :]
                L_Y = L_Y[:, Y]
    return np.array(items)[Y]


def print_unif_samps_dets(L, k, num_samps, rng):
    """
    prints the determinants of num_samps uniform samples
    """
    dets = []
    for i in range(num_samps):
        initial = rng.choice(range(len(L)), size=k, replace=False)
        X = [False] * len(L)
        for j in initial:
            X[j] = True
        X = np.array(X)
        L_Y_cur = L[X,:]
        L_Y_cur = L_Y_cur[:,X]
        dets.append(np.linalg.det(L_Y_cur))
    print("about to sort determinants...")
    sorted_dets = np.sort(dets)
    print(sorted_dets[1:500])
    print(sorted_dets[len(sorted_dets)-500:len(sorted_dets)-1])
    print('avg: {}'.format(np.average(dets)))
     
# returns the determinant of the submatrix of L defined by X
def det_X(X,L):
    L_Y_cur = L[X,:]
    L_Y_cur = L_Y_cur[:,X]
    return np.linalg.det(L_Y_cur)



def get_initial_sample(unif_sampler, k):
    samples = []
    for i in range(k):
        samples.append(unif_sampler.draw_sample())


# returns a sample from a DPP defined over a mixed discrete and contiuous space
# requries something that draws samples uniformly from the space
# requires something that takes two elements and returns a distance between them 
def sample_k_disc_and_cont(unif_sampler, dist_comp, k, max_iter):

    init_sample = get_initial_sample(unif_sampler, k)
    # resample if bad
    # for max iters:
    #  propose new sample
    #  sample within cur sample
    #  compute p
    

    

def sample_discrete_L(L,k,rng,items):
    initial = rng.choice(range(len(items)), size=k, replace=False)
    X = [False] * len(items)
    for i in initial:
        X[i] = True
    X = np.array(X)
    return X


def sample_k(items, L, k, max_nb_iterations=None, rng=np.random):
    """
    Sample a list of k items from a DPP defined
    by the similarity matrix L. The algorithm
    is iterative and runs for max_nb_iterations.
    The algorithm used is from
    (Fast Determinantal Point Process Sampling with
    Application to Clustering, Byungkon Kang, NIPS 2013)
    """
    #import pdb; pdb.set_trace()
    # if L is infinite (some dims of the space are continuous)
    sample_continuous = type(L) == type({})

    print_debug = False

    if max_nb_iterations is None:
        import math
        max_nb_iterations = 5*int(len(L)*math.log(len(L)))
    
    if not sample_continuous:        
        X = sample_discrete_L(L,k,rng,items)
    else:
        initial = sample_continuous_L(L,k)


    # if Y has very close to zero determinant, resample it
    num_Y_resampled = 0
    tolerance = 10**-100
    while det_X(X, L) < tolerance:
        initial = rng.choice(range(len(items)), size=k, replace=False)
        X = [False] * len(items)
        for i in initial:
            X[i] = True
        X = np.array(X)
        num_Y_resampled += 1
        if num_Y_resampled > (1.0/2)*len(L):
            print("We've tried to sample Y such that L_Y is invertible (has det(L_Y) > 0)" + 
                  " but after {} samples we didn't find any with det(L_Y) > {}.".format(
                      (1.0/2)*len(L),tolerance))
            raise ZeroDivisionError("The matrix L is likely low rank => det(L_Y) = 0.")

    if print_debug:
        numerator_counter = 0
        denom_counter = 0
        num_neg_counter = 0
        denom_neg_counter = 0
        p_neg_counter = 0
        both_neg_counter = 0
        
    steps_taken = 0
    num_Y_not_invert = 0
    for i in range(max_nb_iterations):
        
        u = rng.choice(np.arange(len(items))[X])
        v = rng.choice(np.arange(len(items))[~X])
        Y = X.copy()
        Y[u] = False
        L_Y = L[Y, :]
        L_Y = L_Y[:, Y]

        # to check determinants
        if print_debug:
            Y_cur = X.copy()
            L_Y_cur = L[Y_cur,:]
            L_Y_cur = L_Y_cur[:,Y_cur]
            
            Y_next = X.copy()
            Y_next[u] = False
            Y_next[v] = True
            L_Y_next = L[Y_next,:]
            L_Y_next = L_Y_next[:,Y_next]

                  
        try:
            L_Y_inv = np.linalg.inv(L_Y)
        except:
            num_Y_not_invert += 1
            continue
            #import pdb; pdb.set_trace()


        c_v = L[v:v+1, :]
        c_v = c_v[:, v:v+1]
        b_v = L[Y, :]
        b_v = b_v[:, v:v+1]
        c_u = L[u:u+1, :]
        c_u = c_u[:, u:u+1]
        b_u = L[Y, :]
        b_u = b_u[:, u:u+1]


        numerator = c_v - np.dot(np.dot(b_v.T, L_Y_inv.T), b_v)
        denom = c_u - np.dot(np.dot(b_u.T, L_Y_inv.T), b_u)

        if print_debug:
            if numerator < 0 and denom > 0:
                num_neg_counter += 1
            if numerator < 10**-9:
                numerator_counter += 1
            if denom < 0 and numerator > 0:
                denom_neg_counter += 1
            if denom < 10**-9:
                denom_counter += 1
        
            if numerator < 0 and denom < 0:
                both_neg_counter += 1


        p = min(1,  numerator/denom)
        
        # to print if we have some problems with small or zero determinants / eigenvalues
        if print_debug:
            if numerator < 0 or denom < 0 or p < 0:
                print i, p, numerator, denom#u, v, [j for j, b_var in enumerate(Y) if b_var]
                print("{}\t->\t{}".format(np.linalg.det(L_Y_cur), np.linalg.det(L_Y_next)))
                print("steps taken so far: {}, {}%".format(steps_taken, round(100.0*steps_taken/i,3)))
                #import pdb; pdb.set_trace()

        if rng.uniform() <= p:
            steps_taken += 1
            X = Y[:]
            X[v] = True
            
            if print_debug:
                
                print("{}\t->\t{}".format(np.linalg.det(L_Y_cur),np.linalg.det(L_Y_next)))

    if print_debug:
        print("num numerators that would be rounded to zero: {}".format(numerator_counter))
        print("num denoms that would be rounded to zero: {}".format(denom_counter))
        print("num_neg_counter: {}".format(num_neg_counter))
        print("denom_neg_counter: {}".format(denom_neg_counter))
        print("both_neg_counter: {}".format(both_neg_counter))
        print("steps taken: {}".format(steps_taken))
        

    if num_Y_not_invert > .5 * max_nb_iterations:
        print("We've tried to sample Y such that L_Y is invertible (has det(L_Y) > 0)" + 
              " but after {} potential mcmc steps, we found L_Y not invertible {} times.".format(
                  .5 * max_nb_iterations, num_Y_not_invert))
        raise ZeroDivisionError("The matrix L is likely low rank => det(L_Y) = 0.")

    if steps_taken == 0:
        print("We ran the MCMC algorithm for {} steps, but it never accepted a metropolis-hastings " + 
              "proposal, so this is just a uniform sample.".format(steps_taken))
        raise ZeroDivisionError("It's likely the matrix L is bad. The MCMC algorithm failed.")

    print("{} steps taken by mcmc algorithm, out of {} possible steps. {}%".format(steps_taken, 
                                    max_nb_iterations, 100.0*steps_taken/max_nb_iterations))
    return np.array(items)[X]





# taken from dpp.py. it's for debugging.
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
    


# this was taken from dpp.py. it's for debugging
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
