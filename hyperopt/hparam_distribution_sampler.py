import numpy as np
from dpp import scale_and_shift


class Distribution(object):
    def make_zero_one(self, x):
        if x == [] or type(x) == type(np.log([])):
            return 0.0
        else:
            return scale_and_shift(self.a, self.b, 0, 1, x)


class Bernoulli(Distribution):
    def __init__(self, p):
        self.p = p
        self.a = 0
        self.b = 1
                                
    def draw_sample(self):
        return np.random.binomial(1,self.p)


class Uniform(Distribution):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def draw_sample(self):
        return np.random.uniform(self.a, self.b)


class QUniform(Distribution):
    def __init__(self, a, b, q):
        self.a = a
        self.b = b
        self.q = q

    def draw_sample(self):
        draw = np.random.uniform(self.a, self.b)
        return np.round(draw/self.q)*self.q


class LogUniform(Distribution):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def draw_sample(self):
        return np.exp(np.random.uniform(self.a, self.b))

    def make_zero_one(self, x):
        return super(LogUniform, self).make_zero_one(np.log(x))
