import numpy as np
import DPP_Sampler

import matlab

import sys




L = matlab.double(
[[1.0, 0.9342, 0.9412, 0.9269, 0.91, 0.8181, 0.7086, 0.7721, 0.8762, 0.8242, 0.8257, 0.7602, 0.8984, 0.8424, 0.9732, 0.9562, 0.8897, 0.7742, 0.8525, 0.7532, 0.7844, 0.7491, 0.8523, 0.932, 0.8048],
[0.9342, 1.0, 0.8199, 0.8931, 0.8609, 0.7464, 0.7523, 0.9272, 0.8084, 0.8261, 0.7687, 0.8376, 0.739, 0.7034, 0.8853, 0.8551, 0.7311, 0.7237, 0.6701, 0.7358, 0.7359, 0.5122, 0.6811, 0.844, 0.9456],
[0.9412, 0.8199, 1.0, 0.9083, 0.891, 0.8611, 0.4694, 0.6502, 0.8261, 0.6278, 0.8118, 0.6709, 0.9312, 0.8354, 0.9658, 0.9738, 0.8491, 0.7309, 0.9485, 0.6156, 0.7239, 0.7766, 0.8443, 0.7911, 0.6858],
[0.9269, 0.8931, 0.9083, 1.0, 0.957, 0.9413, 0.6797, 0.8166, 0.8699, 0.7826, 0.8216, 0.8522, 0.7422, 0.7581, 0.9637, 0.9659, 0.8374, 0.8436, 0.8257, 0.8363, 0.8187, 0.7391, 0.7181, 0.7653, 0.7836],
[0.91, 0.8609, 0.891, 0.957, 1.0, 0.8986, 0.5845, 0.7957, 0.792, 0.6876, 0.8873, 0.7845, 0.7605, 0.6303, 0.9104, 0.9223, 0.7779, 0.6704, 0.8838, 0.8514, 0.877, 0.7322, 0.7322, 0.7725, 0.7134],
[0.8181, 0.7464, 0.8611, 0.9413, 0.8986, 1.0, 0.5641, 0.7311, 0.6914, 0.6626, 0.647, 0.6569, 0.633, 0.6425, 0.9071, 0.9324, 0.7169, 0.7619, 0.8237, 0.7208, 0.6448, 0.6746, 0.7499, 0.6304, 0.68],
[0.7086, 0.7523, 0.4694, 0.6797, 0.5845, 0.5641, 1.0, 0.6926, 0.6757, 0.9799, 0.4249, 0.6664, 0.3951, 0.6172, 0.6719, 0.6204, 0.6881, 0.7611, 0.3102, 0.7761, 0.5348, 0.4777, 0.5475, 0.7872, 0.7135],
[0.7721, 0.9272, 0.6502, 0.8166, 0.7957, 0.7311, 0.6926, 1.0, 0.5899, 0.7177, 0.6058, 0.7586, 0.4786, 0.4308, 0.7399, 0.7207, 0.4767, 0.5779, 0.5082, 0.6767, 0.5968, 0.257, 0.5286, 0.6447, 0.9587],
[0.8762, 0.8084, 0.8261, 0.8699, 0.792, 0.6914, 0.6757, 0.5899, 1.0, 0.7895, 0.8579, 0.8969, 0.7899, 0.9032, 0.8786, 0.8539, 0.9485, 0.8941, 0.7065, 0.7946, 0.8611, 0.8393, 0.5736, 0.7881, 0.6244],
[0.8242, 0.8261, 0.6278, 0.7826, 0.6876, 0.6626, 0.9799, 0.7177, 0.7895, 1.0, 0.5477, 0.7295, 0.5608, 0.7511, 0.7975, 0.7516, 0.8086, 0.8418, 0.4723, 0.8058, 0.6248, 0.6081, 0.6597, 0.8681, 0.7551],
[0.8257, 0.7687, 0.8118, 0.8216, 0.8873, 0.647, 0.4249, 0.6058, 0.8579, 0.5477, 1.0, 0.8319, 0.7965, 0.6402, 0.7821, 0.7821, 0.7867, 0.5839, 0.8116, 0.7912, 0.9697, 0.7668, 0.5228, 0.7097, 0.5407],
[0.7602, 0.8376, 0.6709, 0.8522, 0.7845, 0.6569, 0.6664, 0.7586, 0.8969, 0.7295, 0.8319, 1.0, 0.5493, 0.6582, 0.7554, 0.7309, 0.735, 0.8016, 0.5266, 0.8354, 0.8639, 0.5929, 0.3256, 0.6142, 0.7197],
[0.8984, 0.739, 0.9312, 0.7422, 0.7605, 0.633, 0.3951, 0.4786, 0.7899, 0.5608, 0.7965, 0.5493, 1.0, 0.8424, 0.8627, 0.8554, 0.8431, 0.5928, 0.8978, 0.4981, 0.6837, 0.7717, 0.8329, 0.842, 0.5607],
[0.8424, 0.7034, 0.8354, 0.7581, 0.6303, 0.6425, 0.6172, 0.4308, 0.9032, 0.7511, 0.6402, 0.6582, 0.8424, 1.0, 0.869, 0.8418, 0.9374, 0.8865, 0.6927, 0.5481, 0.5999, 0.8077, 0.7115, 0.7914, 0.565],
[0.9732, 0.8853, 0.9658, 0.9637, 0.9104, 0.9071, 0.6719, 0.7399, 0.8786, 0.7975, 0.7821, 0.7554, 0.8627, 0.869, 1.0, 0.9957, 0.8983, 0.8492, 0.8752, 0.7374, 0.7456, 0.7841, 0.8458, 0.8526, 0.7702],
[0.9562, 0.8551, 0.9738, 0.9659, 0.9223, 0.9324, 0.6204, 0.7207, 0.8539, 0.7516, 0.7821, 0.7309, 0.8554, 0.8418, 0.9957, 1.0, 0.8804, 0.8267, 0.9041, 0.7275, 0.7422, 0.7924, 0.8474, 0.8173, 0.7367],
[0.8897, 0.7311, 0.8491, 0.8374, 0.7779, 0.7169, 0.6881, 0.4767, 0.9485, 0.8086, 0.7867, 0.735, 0.8431, 0.9374, 0.8983, 0.8804, 1.0, 0.8696, 0.7799, 0.7703, 0.7967, 0.9351, 0.7379, 0.8629, 0.524],
[0.7742, 0.7237, 0.7309, 0.8436, 0.6704, 0.7619, 0.7611, 0.5779, 0.8941, 0.8418, 0.5839, 0.8016, 0.5928, 0.8865, 0.8492, 0.8267, 0.8696, 1.0, 0.557, 0.7122, 0.6242, 0.7296, 0.5534, 0.6675, 0.6386],
[0.8525, 0.6701, 0.9485, 0.8257, 0.8838, 0.8237, 0.3102, 0.5082, 0.7065, 0.4723, 0.8116, 0.5266, 0.8978, 0.6927, 0.8752, 0.9041, 0.7799, 0.557, 1.0, 0.5954, 0.7298, 0.8109, 0.8373, 0.7197, 0.4879],
[0.7532, 0.7358, 0.6156, 0.8363, 0.8514, 0.7208, 0.7761, 0.6767, 0.7946, 0.8058, 0.7912, 0.8354, 0.4981, 0.5481, 0.7374, 0.7275, 0.7703, 0.7122, 0.5954, 1.0, 0.9014, 0.7306, 0.4914, 0.7201, 0.5585],
[0.7844, 0.7359, 0.7239, 0.8187, 0.877, 0.6448, 0.5348, 0.5968, 0.8611, 0.6248, 0.9697, 0.8639, 0.6837, 0.5999, 0.7456, 0.7422, 0.7967, 0.6242, 0.7298, 0.9014, 1.0, 0.79, 0.4633, 0.7016, 0.498],
[0.7491, 0.5122, 0.7766, 0.7391, 0.7322, 0.6746, 0.4777, 0.257, 0.8393, 0.6081, 0.7668, 0.5929, 0.7717, 0.8077, 0.7841, 0.7924, 0.9351, 0.7296, 0.8109, 0.7306, 0.79, 1.0, 0.6658, 0.7149, 0.2485],
[0.8523, 0.6811, 0.8443, 0.7181, 0.7322, 0.7499, 0.5475, 0.5286, 0.5736, 0.6597, 0.5228, 0.3256, 0.8329, 0.7115, 0.8458, 0.8474, 0.7379, 0.5534, 0.8373, 0.4914, 0.4633, 0.6658, 1.0, 0.8566, 0.5882],
[0.932, 0.844, 0.7911, 0.7653, 0.7725, 0.6304, 0.7872, 0.6447, 0.7881, 0.8681, 0.7097, 0.6142, 0.842, 0.7914, 0.8526, 0.8173, 0.8629, 0.6675, 0.7197, 0.7201, 0.7016, 0.7149, 0.8566, 1.0, 0.6978],
[0.8048, 0.9456, 0.6858, 0.7836, 0.7134, 0.68, 0.7135, 0.9587, 0.6244, 0.7551, 0.5407, 0.7197, 0.5607, 0.565, 0.7702, 0.7367, 0.524, 0.6386, 0.4879, 0.5585, 0.498, 0.2485, 0.5882, 0.6978, 1.0]])







#L = matlab.double([[1,.8,.6],[.8,1,.7],[.6,.7,1]])
dpp = DPP_Sampler.initialize()
L_decomp = dpp.decompose_kernel(L)
dpp_samples = dpp.sample_dpp(L_decomp, 939857, 2)
print(dpp_samples)
dpp.terminate()


sys.exit(0)
r = np.random.rand(250,6)
for i in range(len(r)): r[i] = r[i]/np.linalg.norm(r[i],2)

print("normalized r")
L_np = np.dot(r,np.transpose(r))
print("made L_np")
L_list = np.ndarray.tolist(L_np)
# if we want to round:
for i in range(len(L_list)): L_list[i] = [round(L_list[i][j],4) for j in range(len(L_list[i]))]

print("made L_list")
L = matlab.double(L_list)
print("created L!")
sys.stdout.flush()
L_decomp = dpp.decompose_kernel(L)
dpp_samples = dpp.sample_dpp(L_decomp, 93857, 2)
print(dpp_samples)
