import matlab
import DPP_Sampler
import numpy as np


# returns a matlab.double array which contains ints.
def sample_dpp(L_np, seed, k):

    
    L_list = np.ndarray.tolist(L_np)

    #L_list=[[1,.8,.7],[.8,1,.8],[.7,.8,1]]

    L = matlab.double(L_list)
    dpp_samp = DPP_Sampler.initialize()
    L_decomp = dpp_samp.decompose_kernel(L)
    dpp_samples = dpp_samp.sample_dpp(L_decomp,seed,k)
    dpp_samp.terminate()
    return dpp_samples



