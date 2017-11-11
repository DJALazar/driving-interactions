import theano as th
import theano.tensor as tt

"""
nx is number of elements in the x-vector (car movement state information)
nu is the number of elements in the u-vector (car control input)
"""

class Dynamics(object):
    def __init__(self, nx, nu, f, dt=None):
        self.nx = nx
        self.nu = nu
        self.dt = dt
        if dt is None:
            self.f = f
        else:
            self.f = lambda x, u: x+dt*f(x, u)
    def __call__(self, x, u):
        return self.f(x, u)

"""
x[0] is location in x direction
x[1] is location in y direction
x[2] is angle of heading
x[3] is speed

dt is time step

output of f is element-wise change in x. 
multiply output of f by dt and add to x for new x

u is the input.
u[0] is steering
u[1] is acceleration
"""
class CarDynamics(Dynamics):
    def __init__(self, dt=0.1, ub=[(-3., 3.), (-1., 1.)], friction=1.):
        def f(x, u):
            return tt.stacklists([
                x[3]*tt.cos(x[2]),
                x[3]*tt.sin(x[2]),
                x[3]*u[0],
                u[1]-x[3]*friction
            ])
        Dynamics.__init__(self, 4, 2, f, dt)

if __name__ == '__main__':
    dyn = CarDynamics(0.1)
    x = tt.vector()
    u = tt.vector()
    dyn(x, u)
