import lane
import car
import math
import feature
import dynamics
import visualize
import utils
import sys
import theano as th
import theano.tensor as tt
import numpy as np
import shelve

th.config.optimizer_verbose = True
th.config.allow_gc = False
th.config.optimizer = 'fast_compile'

class Object(object):
    def __init__(self, name, x):
        self.name = name
        self.x = np.asarray(x)

class World(object):
    def __init__(self):
        self.cars = []
        self.lanes = []
        self.roads = []
        self.fences = []
        self.objects = []
    def simple_reward(self, trajs=None, lanes=None, roads=None, fences=None, speed=1., speed_import=1.):
        if lanes is None:
            lanes = self.lanes
        if roads is None:
            roads = self.roads
        if fences is None:
            fences = self.fences
        if trajs is None:
            trajs = [c.linear for c in self.cars]
        elif isinstance(trajs, car.Car):
            trajs = [c.linear for c in self.cars if c!=trajs]
        r = 0.1*feature.control()
        """
        # What is theta? First one is importance of staying in lanes, 
        second is staying on the road entirely (not violating the outer fence)
        third is staying on the road also?
        fourth is maintaining desired speed
        fifth is ...?
        """
        theta = [1., -50., 10., 10., -60.] # Simple model
        # theta = [.959, -46.271, 9.015, 8.531, -57.604]
        for lane in lanes:
            r = r+theta[0]*lane.gaussian()
        for fence in fences:
            r = r+theta[1]*fence.gaussian()
        for road in roads:
            r = r+theta[2]*road.gaussian(10.)
        if speed is not None:
            r = r+speed_import*theta[3]*feature.speed(speed)
        for traj in trajs:
            r = r+theta[4]*traj.gaussian()
        return r

def playground():
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.17)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2)]
    #world.cars.append(car.UserControlledCar(dyn, [0., 0., math.pi/2., 0.], color='orange'))
    world.cars.append(car.UserControlledCar(dyn, [-0.17, -0.17, math.pi/2., 0.], color='white'))
    return world

def irl_ground():
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2)]
    d = shelve.open('cache', writeback=True)
    cars = [(-.13, .1, .5, 0.13),
            (.02, .4, .8, 0.5),
            (.13, .1, .6, .13),
            (-.09, .8, .5, 0.),
            (0., 1., 0.5, 0.),
            (-.13, -0.5, 0.9, 0.13),
            (.13, -.8, 1., -0.13),
           ]
    def goal(g):
        @feature.feature
        def r(t, x, u):
            return -(x[0]-g)**2
        return r
    for i, (x, y, s, gx) in enumerate(cars):
        if str(i) not in d:
            d[str(i)] = []
        world.cars.append(car.SimpleOptimizerCar(dyn, [x, y, math.pi/2., s], color='yellow'))
        world.cars[-1].cache = d[str(i)]
        def f(j):
            def sync(cache):
                d[str(j)] = cache
                d.sync()
            return sync
        world.cars[-1].sync = f(i)
    for c, (x, y, s, gx) in zip(world.cars, cars):
        c.reward = world.simple_reward(c, speed=s)+10.*goal(gx)
    world.cars.append(car.UserControlledCar(dyn, [0., 0., math.pi/2., 0.7], color='red'))
    world.cars = world.cars[-1:]+world.cars[:-1]
    return world

def world_test():
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2)]
    world.cars.append(car.UserControlledCar(dyn, [-0.13, 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.0, 0.5, math.pi/2., 0.3], color='yellow'))
    world.cars[1].reward = world.simple_reward(world.cars[1], speed=0.5)
    return world

def world0():
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2)]
    world.cars.append(car.UserControlledCar(dyn, [-0.13, 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.NestedOptimizerCar(dyn, [0.0, 0.5, math.pi/2., 0.3], color='yellow'))
    world.cars[1].human = world.cars[0]
    r_h = world.simple_reward([world.cars[1].traj])+100.*feature.bounded_control(world.cars[0].bounds)
    @feature.feature
    def human_speed(t, x, u):
        return -world.cars[1].traj_h.x[t][3]**2
    r_r = world.simple_reward(world.cars[1], speed=0.5)
    world.cars[1].rewards = (r_h, r_r)
    return world

"""
In this world, the robot car tries to get the human to slow down
"""
def world1(flag=False):
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2)]
    world.cars.append(car.UserControlledCar(dyn, [-0.13, 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.NestedOptimizerCar(dyn, [0.0, 0.5, math.pi/2., 0.3], color='yellow'))
    world.cars[1].human = world.cars[0]
    if flag:
        world.cars[0].follow = world.cars[1].traj_h
    r_h = world.simple_reward([world.cars[1].traj], speed_import=.2 if flag else 1., speed=0.8 if flag else 1.)+100.*feature.bounded_control(world.cars[0].bounds)
    @feature.feature
    def human_speed(t, x, u):
        return -world.cars[1].traj_h.x[t][3]**2
    r_r = 300.*human_speed+world.simple_reward(world.cars[1], speed=0.5)
    if flag:
        world.cars[0].follow = world.cars[1].traj_h
    world.cars[1].rewards = (r_h, r_r)
    #world.objects.append(Object('cone', [0., 1.8]))
    return world

"""
In this world, the robot car tries to get the human to go into the first lane
"""
def world2(flag=False):
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2), clane.shifted(2.5), clane.shifted(-2.5)]
    world.cars.append(car.UserControlledCar(dyn, [0., 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.NestedOptimizerCar(dyn, [0., 0.3, math.pi/2., 0.3], color='yellow'))
    world.cars[1].human = world.cars[0]
    world.cars[0].bounds = [(-3., 3.), (-1., 1.)]
    if flag:
        world.cars[0].follow = world.cars[1].traj_h
    r_h = world.simple_reward([world.cars[1].traj])+100.*feature.bounded_control(world.cars[0].bounds)
    @feature.feature
    def human(t, x, u):
        return -(world.cars[1].traj_h.x[t][0])*10
    r_r = 300.*human+world.simple_reward(world.cars[1], speed=0.5)
    world.cars[1].rewards = (r_h, r_r)
    #world.objects.append(Object('firetruck', [0., 0.7]))
    return world

"""
Same as last wold, except reward is negative, so robot car tries to get human to go away from the first lane
"""
def world3(flag=False):
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2), clane.shifted(2.5), clane.shifted(-2.5)]
    world.cars.append(car.UserControlledCar(dyn, [0., 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.NestedOptimizerCar(dyn, [0., 0.3, math.pi/2., 0.3], color='yellow'))
    world.cars[1].human = world.cars[0]
    world.cars[0].bounds = [(-3., 3.), (-1., 1.)]
    if flag:
        world.cars[0].follow = world.cars[1].traj_h
    r_h = world.simple_reward([world.cars[1].traj])+100.*feature.bounded_control(world.cars[0].bounds)
    @feature.feature
    def human(t, x, u):
        return (world.cars[1].traj_h.x[t][0])*10
    r_r = 300.*human+world.simple_reward(world.cars[1], speed=0.5)
    world.cars[1].rewards = (r_h, r_r)
    #world.objects.append(Object('firetruck', [0., 0.7]))
    return world

"""
This world is an intersection, and the robot car tries to have the human driver cross an intersection
"""
def world4(flag=False):
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    vlane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    hlane = lane.StraightLane([-1., 0.], [1., 0.], 0.13)
    world.lanes += [vlane, hlane]
    world.fences += [hlane.shifted(-1), hlane.shifted(1)]
    world.cars.append(car.UserControlledCar(dyn, [0., -.3, math.pi/2., 0.0], color='red'))
    world.cars.append(car.NestedOptimizerCar(dyn, [-0.3, 0., 0., 0.], color='yellow'))
    world.cars[1].human = world.cars[0]
    world.cars[0].bounds = [(-3., 3.), (-2., 2.)]
    if flag:
        world.cars[0].follow = world.cars[1].traj_h
    world.cars[1].bounds = [(-3., 3.), (-2., 2.)]
    @feature.feature
    def horizontal(t, x, u):
        # isn't x[2] the heading, not the y location?
        return -x[2]**2
    r_h = world.simple_reward([world.cars[1].traj], lanes=[vlane], fences=[vlane.shifted(-1), vlane.shifted(1)]*2)+100.*feature.bounded_control(world.cars[0].bounds)
    @feature.feature
    def human(t, x, u):
        return -tt.exp(-10*(world.cars[1].traj_h.x[t][1]-0.13)/0.1)
    r_r = human*10.+horizontal*30.+world.simple_reward(world.cars[1], lanes=[hlane]*3, fences=[hlane.shifted(-1), hlane.shifted(1)]*3+[hlane.shifted(-1.5), hlane.shifted(1.5)]*2, speed=0.9)
    world.cars[1].rewards = (r_h, r_r)
    return world

def world5():
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    vlane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    hlane = lane.StraightLane([-1., 0.], [1., 0.], 0.13)
    world.lanes += [vlane, hlane]
    world.fences += [hlane.shifted(-1), hlane.shifted(1)]
    world.cars.append(car.UserControlledCar(dyn, [0., -.3, math.pi/2., 0.0], color='red'))
    world.cars.append(car.NestedOptimizerCar(dyn, [-0.3, 0., 0., 0.0], color='yellow'))
    world.cars[1].human = world.cars[0]
    world.cars[1].bounds = [(-3., 3.), (-2., 2.)]
    @feature.feature
    def horizontal(t, x, u):
        return -x[2]**2
    r_h = world.simple_reward([world.cars[1].traj], lanes=[vlane], fences=[vlane.shifted(-1), vlane.shifted(1)]*2)+100.*feature.bounded_control(world.cars[0].bounds)
    @feature.feature
    def human(t, x, u):
        return -tt.exp(10*(world.cars[1].traj_h.x[t][1]-0.13)/0.1)
    r_r = human*10.+horizontal*2.+world.simple_reward(world.cars[1], lanes=[hlane]*3, fences=[hlane.shifted(-1), hlane.shifted(1)]*3+[hlane.shifted(-1.5), hlane.shifted(1.5)]*2, speed=0.9)
    world.cars[1].rewards = (r_h, r_r)
    return world


def world6(know_model=False):
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2), clane.shifted(2.5), clane.shifted(-2.5)]
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0., math.pi/2., 0.5], color='red'))
    if know_model:
        world.cars.append(car.NestedOptimizerCar(dyn, [0., 0.05, math.pi/2., 0.5], color='yellow'))
    else:
        world.cars.append(car.SimpleOptimizerCar(dyn, [0., 0.05, math.pi/2., 0.5], color='yellow'))
    world.cars[0].reward = world.simple_reward(world.cars[0], speed=0.6)
    world.cars[0].default_u = np.asarray([0., 1.])
    @feature.feature
    def goal(t, x, u):
        return -(10.*(x[0]+0.13)**2+0.5*(x[1]-2.)**2)
    if know_model:
        world.cars[1].human = world.cars[0]
        r_h = world.simple_reward([world.cars[1].traj], speed=0.6)+100.*feature.bounded_control(world.cars[0].bounds)
        r_r = 10*goal+world.simple_reward([world.cars[1].traj_h], speed=0.5)
        world.cars[1].rewards = (r_h, r_r)
    else:
        r = 10*goal+world.simple_reward([world.cars[0].linear], speed=0.5)
        world.cars[1].reward = r
    return world

"""
This world has 13 simple optimizers in it
"""
def world7():
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2), clane.shifted(2.5), clane.shifted(-2.5)]

    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0.6, math.pi/2,0.5], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0.3, math.pi/2,0.5], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0., math.pi/2., 0.5], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, -0.3, math.pi/2,0.5], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, -0.6, math.pi/2,0.5], color='red'))    
    world.cars.append(car.SimpleOptimizerCar(dyn, [0., 0.3, math.pi/2,0.5], color='orange'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0., 0.05, math.pi/2., 0.5], color='orange'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0., -0.3, math.pi/2,0.5], color='orange'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0., -0.6, math.pi/2,0.5], color='orange'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.13, 0.3, math.pi/2,0.5], color='yellow'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.13, 0., math.pi/2,0.5], color='yellow'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.13, -0.3, math.pi/2,0.5], color='yellow'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.13, -0.6, math.pi/2,0.5], color='yellow'))
    world.cars[0].reward = world.simple_reward(world.cars[0], speed=0.5)
    world.cars[0].default_u = np.asarray([0., 1.])
    world.cars[1].reward = world.simple_reward(world.cars[1], speed=0.55)
    world.cars[1].default_u = np.asarray([0., 1.])
    world.cars[2].reward = world.simple_reward(world.cars[2], speed=0.5)
    world.cars[2].default_u = np.asarray([0., 1.])
    world.cars[3].reward = world.simple_reward(world.cars[3], speed=1.5)
    world.cars[3].default_u = np.asarray([0., 1.])
    world.cars[4].reward = world.simple_reward(world.cars[4], speed=0.45)
    world.cars[4].default_u = np.asarray([0., 1.])
    world.cars[5].reward = world.simple_reward(world.cars[5], speed=0.5)
    world.cars[5].default_u = np.asarray([0., 1.])
    world.cars[6].reward = world.simple_reward(world.cars[6], speed=0.5)
    world.cars[6].default_u = np.asarray([0., 1.])
    world.cars[7].reward = world.simple_reward(world.cars[7], speed=0.5)
    world.cars[7].default_u = np.asarray([0., 1.])
    world.cars[8].reward = world.simple_reward(world.cars[8], speed=0.55)
    world.cars[8].default_u = np.asarray([0., 1.])
    world.cars[9].reward = world.simple_reward(world.cars[9], speed=0.5)
    world.cars[9].default_u = np.asarray([0., 1.])
    world.cars[10].reward = world.simple_reward(world.cars[10], speed=0.5)
    world.cars[10].default_u = np.asarray([0., 1.])
    world.cars[11].reward = world.simple_reward(world.cars[11], speed=0.45)
    world.cars[11].default_u = np.asarray([0., 1.])
    world.cars[12].reward = world.simple_reward(world.cars[12], speed=0.5)
    world.cars[12].default_u = np.asarray([0., 1.])
    return world   

"""
Broken world, something to do with the caching
"""
def world8():
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2)]
    d = shelve.open('cache', writeback=True)
    """
    cars = [(-.13, .1, 0.5, 0.13),
            (.02, .4, 0.5, 0.5),
            (.13, .1, 0.5, .13),
            (-.09, .8, 0.5, 0.),
            (0., 1., 0.5, 0.),
            (-.13, -0.5, 0.5, 0.13),
            (.13, -.8, 0.5, -0.13),
            (-.13, .5, 0.5, 0.13),
            (.13, 0.7, 0.5, -0.13),
           ]
    """
    cars = [(-0.13, 0.1, 0.5, -0.13),
            (.02, 0.4, 0.5, 0.0),
            (.13, 0.1, 0.5, .13),
            (-.09, .8, 0.5, 0.0),
            (0.0, 1.0, 0.5, 0.0),
            (0.13, -0.5, 0.5, 0.13),
            (0.13, -0.8, 0.5, 0.13),
            (-0.13, 0.5, 0.5, -0.13),
            (0.13, 0.7, 0.5, 0.13),
           ]
    def goal(g):
        @feature.feature
        def r(t, x, u):
            return -(x[0]-g)**2
        return r
    for i, (x, y, s, gx) in enumerate(cars):
        if str(i) not in d:
            d[str(i)] = []
        world.cars.append(car.SimpleOptimizerCar(dyn, [x, y, math.pi/2., s], color='yellow'))
        world.cars[-1].cache = d[str(i)]
        def f(j):
            def sync(cache):
                d[str(j)] = cache
                d.sync()
            return sync
        world.cars[-1].sync = f(i)
    for c, (x, y, s, gx) in zip(world.cars, cars):
        c.reward = world.simple_reward(c, speed=s)+10.*goal(gx)
        #c.reward = world.simple_reward(c, speed=s)
    #world.cars.append(car.UserControlledCar(dyn, [0., 0., math.pi/2., 0.7], color='red'))
    #world.cars = world.cars[-1:]+world.cars[:-1]
    return world

"""
The goal of this world is to get cars to platoon. Start with one vehicle a distance behind the other, 
and give it the goal to get within a certain distance of the other. Once it gets to that distance,
have them platoon.

In this world, all cars are simple optimizers, and there is one car between the cars that wish to platoon
"""
def world9():

    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2), clane.shifted(2.5), clane.shifted(-2.5)]

    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0., math.pi/2,0.5], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, -1.3, math.pi/2,0.5], color='yellow'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, -0.5, math.pi/2,0.5], color='orange'))

    world.cars[0].reward = world.simple_reward(world.cars[0], speed=0.5)
    world.cars[1].reward = world.simple_reward(world.cars[1], speed=0.5) + 100*veh_follow
    world.cars[2].reward = world.simple_reward(world.cars[2], speed=0.5)

    return world

"""
This is the same as the previous world, except that the car that is trying to platoon behind a car in front is a nested optimizer

LOOK WHAT WE MADE THE ROBOTS DO
"""
def world10():

    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2), clane.shifted(2.5), clane.shifted(-2.5)]

    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0., math.pi/2,0.5], color='red'))
    world.cars.append(car.NestedOptimizerCar(dyn, [-0.13, -1.3, math.pi/2,0.5], color='yellow'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, -0.3, math.pi/2,0.5], color='orange'))

    world.cars[1].human = world.cars[2]
    #r_h = world.simple_reward([world.cars[1].traj])+100.*feature.bounded_control(world.cars[0].bounds)

    @feature.feature
    def veh_follow(t, x, u):
        return -((world.cars[0].traj.x[t][0]-world.cars[1].traj.x[t][0])**4 + (world.cars[0].traj.x[t][1]-0.3-world.cars[1].traj.x[t][1])**2)

    r_r = world.simple_reward(world.cars[1], speed=0.5) + 100*veh_follow
    r_h = world.simple_reward(world.cars[2], speed=0.5)   

    world.cars[0].reward = world.simple_reward(world.cars[0], speed=0.5)
    world.cars[1].rewards = (r_h, r_r)
    world.cars[2].reward = r_h

    return world

"""
Try the nested optimizer with different goal functions
"""

def world11():

    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2), clane.shifted(2.5), clane.shifted(-2.5)]

    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0., math.pi/2,0.5], color='red'))
    world.cars.append(car.NestedOptimizerCar(dyn, [-0.13, -1.3, math.pi/2,0.5], color='yellow'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, -0.5, math.pi/2,0.5], color='orange'))

    world.cars[1].human = world.cars[2]
    #r_h = world.simple_reward([world.cars[1].traj])+100.*feature.bounded_control(world.cars[0].bounds)

    """
    This goal just tries to get the cars close to each other
    """
    """
    @feature.feature
    def veh_follow(t, x, u):
        return -((world.cars[0].traj.x[t][0]-world.cars[1].traj.x[t][0])**2 + (world.cars[0].traj.x[t][1]-0.3-world.cars[1].traj.x[t][1])**2)
    """

    """
    For the first few time steps, try to get the vehicles close together in the y-dimension.
    After a number of steps, get them together in the x-dimension. This allows for strategic lane switching.
    """
    """
    @feature.feature
    def veh_follow(t, x, u):
        if (t>3):
            s = -(5*(world.cars[0].traj.x[t][0]-world.cars[1].traj.x[t][0])**2 + (world.cars[0].traj.x[t][1]-0.3-world.cars[1].traj.x[t][1])**2)
        else :
            s = -((world.cars[0].traj.x[t][1]-0.3-world.cars[1].traj.x[t][1])**2);
        return s
    """

    """
    This goal ramps up how much the x-dimension matters as the vehicle gets closer to platooning position. Also, the y-position goal saturates.
    """
    @feature.feature
    def veh_follow(t, x, u):

        follow_loc = 0.3

        distance_sq = (world.cars[0].traj.x[t][0]-world.cars[1].traj.x[t][0])**2 + (world.cars[0].traj.x[t][1]-world.cars[1].traj.x[t][1])**2

        distance = np.sqrt(distance_sq)

        # if we are not close to the goal car, it doesn't matter what lane we're in (hence the exp term -- importance of being in the correct lane decays exponentially with y-distance from target)
        x_penalty = -10*tt.exp(-10.0*(distance-follow_loc))*(world.cars[0].traj.x[t][0]-world.cars[1].traj.x[t][0])**2

        # The y penalty should saturate at a certain distance because a very distant car shouldn't engage in very risky maneuvers.
        # Because of this we have the exponential saturation term.
        #y_penalty = -(world.cars[0].traj.x[t][1]-follow_loc-world.cars[1].traj.x[t][1])**2
        y_penalty = -(-1.0/2.0 + 100.0/(1.0 + tt.exp(-1.0/10.0*(world.cars[0].traj.x[t][1] - world.cars[1].traj.x[t][1] - follow_loc)**2) ) )

        s = x_penalty + y_penalty
        
        return s


    r_r = world.simple_reward(world.cars[1], speed=0.5) + 100*veh_follow
    r_h = world.simple_reward(world.cars[2], speed=0.5)

    world.cars[0].reward = world.simple_reward(world.cars[0], speed=0.5)
    world.cars[1].rewards = (r_h, r_r)
    world.cars[2].reward = r_h

    return world 

def world_features(num=0):
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2)]
    world.cars.append(car.UserControlledCar(dyn, [-0.13, 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.Car(dyn, [0., 0.1, math.pi/2.+math.pi/5, 0.], color='yellow'))
    world.cars.append(car.Car(dyn, [-0.13, 0.2, math.pi/2.-math.pi/5, 0.], color='yellow'))
    world.cars.append(car.Car(dyn, [0.13, -0.2, math.pi/2., 0.], color='yellow'))
    #world.cars.append(car.NestedOptimizerCar(dyn, [0.0, 0.5, math.pi/2., 0.3], color='yellow'))
    return world

if __name__ == '__main__':
    world = playground()
    #world.cars = world.cars[:0]
    vis = visualize.Visualizer(0.1, magnify=1.2)
    vis.main_car = None
    vis.use_world(world)
    vis.paused = True
    @feature.feature
    def zero(t, x, u):
        return 0.
    r = zero
    #for lane in world.lanes:
    #    r = r+lane.gaussian()
    #for fence in world.fences:
    #    r = r-3.*fence.gaussian()
    r = r - world.cars[0].linear.gaussian()
    #vis.visible_cars = [world.cars[0]]
    vis.set_heat(r)
    vis.run()
