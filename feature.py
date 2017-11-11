import theano as th
import theano.tensor as tt

"""
Define the features in a simple optimizer, such as desired speed, not oversteering, etc.
"""

class Feature(object):
    def __init__(self, f):
        self.f = f
    def __call__(self, *args):
        return self.f(*args)
    def __add__(self, r):
        return Feature(lambda *args: self(*args)+r(*args))
    def __radd__(self, r):
        return Feature(lambda *args: r(*args)+self(*args))
    def __mul__(self, r):
        return Feature(lambda *args: self(*args)*r)
    def __rmul__(self, r):
        return Feature(lambda *args: r*self(*args))
    def __pos__(self, r):
        return self
    def __neg__(self):
        return Feature(lambda *args: -self(*args))
    def __sub__(self, r):
        return Feature(lambda *args: self(*args)-r(*args))
    def __rsub__(self, r):
        return Feature(lambda *args: r(*args)-self(*args))

def feature(f):
    return Feature(f)

# goal defined by a vehicles desired speed
def speed(s=1.):
    @feature
    def f(t, x, u):
        return -(x[3]-s)*(x[3]-s)
    return f

# goal of not hitting acceleration/deceleration or steering too hard
def control():
    @feature
    def f(t, x, u):
        return -u[0]**2-u[1]**2
    return f

#bounds on steering and gas/brake
def bounded_control(bounds, width=0.05):
    @feature
    def f(t, x, u):
        ret = 0.
        for i, (a, b) in enumerate(bounds):
            return -tt.exp((u[i]-b)/width)-tt.exp((a-u[i])/width)
    return f

# Can we put the platooning objective here?
"""
def car_follow(veh1, veh2):
    @feature   
    def f(t, x, u):
"""
         

if __name__ == '__main__':
    pass
