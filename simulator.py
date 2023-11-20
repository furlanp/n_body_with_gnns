import numpy as np
import numpy.linalg as la
from collections import deque
from drawer import PygApp

class Body:    
    # initialization
    def __init__(self, pos, vel, mass):
        # pos and vel can be 2D or 3D numpy arrays 
        self.position = pos
        self.velocity = vel
        self.mass = mass
        # initial acceleration is zero
        self.acceleration = np.zeros(pos.shape[0], dtype=float)

    def update(self, acc, dt):
        self.acceleration = acc
        self.velocity += acc * dt
        self.position += self.velocity * dt

class System:
    def __init__(self, bodies_data, dim):
        ''' bodies_data is passed as a list of dictionaries
            which describes system's initial state 
            
            example of a dictionary (for 3D system) would be:
            body_data = {
                'pos': [-1, 0, 0],
                'vel': [0, 0.1, 0.1],
                'mass': 1000.0
            }
        '''

        self.bodies_data = bodies_data
        self.dim = dim
        self.system_restart()

    # sets the system according to initial state
    def system_restart(self):
        self.bodies = []               
        for body_data in self.bodies_data:
            body_pos = np.array(body_data['pos'], dtype=float)
            body_vel = np.array(body_data['vel'], dtype=float)
            self.bodies.append(Body(body_pos, body_vel, body_data['mass']))

class SunEarthSystem(System):
    def __init__(self):
        earth_data = {
            'pos': [-1, 0],
            'vel': [0, 0.1],
            'mass': 1.0
        }

        sun_data = {
            'pos': [0, 0],
            'vel': [0, 0],
            'mass': 100000.0       
        }

        super().__init__([earth_data, sun_data], 2)

class RK4:    
    def __init__(self, bodies, dt, dim):
        self.bodies = bodies
        self.dt = dt
        self.dim = dim   # 2D or 3D

    def single_body_acc(self, body_idx):
        G = 1e-6   # gravitational constant
        acc = np.zeros(self.dim, dtype=float)

        # body for which we are calculating acceleration
        target = self.bodies[body_idx]
        
        for i, other in enumerate(self.bodies):
            if target == other:
                continue

            C = G * other.mass   # constant
            EPS = 1   # smoothing factor

            k1r = target.velocity
            r = other.position - target.position 
            k1v = C * r / (la.norm(r) + EPS) ** 3

            k2r = target.velocity + k1v * 0.5 * self.dt
            r = other.position - (target.position + k1r * 0.5 * self.dt)
            k2v = C * r / (la.norm(r) + EPS) ** 3

            k3r = target.velocity + k2v * 0.5 * self.dt
            r = other.position - (target.position + k2r * 0.5 * self.dt)
            k3v = C * r / (la.norm(r) + EPS) ** 3

            k4r = target.velocity + k3v * self.dt
            r = other.position - (target.position + k3r * self.dt)
            k4v = C * r / (la.norm(r) + EPS) ** 3

            acc += (2 * k1v + k2v + k3v + 2 * k4v) / 6

        return acc
    
    # updates bodies positions and velocities
    def update_system(self):
        for body_idx, target in enumerate(self.bodies):
            acc = self.single_body_acc(body_idx)
            target.update(acc, self.dt)

# creates dataset for the given System object
def make_dataset(system, include_mass=True, relative=True, no_vel=5, dt=0.1, no_steps=500):
    ''' args:
            system - given System object
            include_mass - whether we include body mass as a feature
            relative - are we storing absolute values or differences
            no_vel - how many past velocities we include as features
            dt - resolution of our computation
            no_steps - how many steps of computation are we performing
    '''

    # restart system state
    system.system_restart()
    bodies = system.bodies

    simulator = RK4(bodies, dt, system.dim)

    dataset = []   # contains particle data for each timestep

    # we store last no_vel velocities for each particle
    velocities = [deque([np.zeros(system.dim)]) for i in range(len(bodies))]
    
    # we need those for calculating differences
    last_pos = [np.zeros(system.dim) for i in range(len(bodies))]

    for i in range(no_steps):
        body_data = []
        
        for body_idx, target in enumerate(bodies):
            # add new velocity
            velocities[body_idx].append(np.copy(target.velocity))
            
            if len(velocities[body_idx]) <= no_vel:   # skip if not enough data
                continue

            curr_body_data = []

            # mass...
            if include_mass:   
                curr_body_data.append(target.mass)

            # position...
            new_pos = np.copy(target.position)
            if relative:
                new_pos -= last_pos[body_idx]
            curr_body_data.extend(list(new_pos)) 
            
            # velocities...
            for i in range(1, no_vel + 1):
                new_vel = velocities[body_idx][i]
                if relative:
                    new_vel -= velocities[body_idx][i - 1]
                curr_body_data.extend(list(new_vel))

            # acceleration...
            curr_body_data.extend(list(target.acceleration))
            
            # prepare for next step
            last_pos[body_idx] = np.copy(target.position)
            velocities[body_idx].popleft()

            body_data.append(curr_body_data)
        
        if len(body_data):
            dataset.append(body_data)

        simulator.update_system()
    
    return np.array(dataset)

if __name__ == '__main__':
    # toy example
    system = SunEarthSystem()

    dataset = make_dataset(system, include_mass=False, relative=False)
    print(dataset.shape)

    N = dataset.shape[0]
    t = 0

    # visualization
    def update_func():
        global t
        points = [list(dataset[t, 0, :2]) + [0], list(dataset[t, 1, :2]) + [0]] 
        t = (t + 1) % N
        return points 
    
    app = PygApp(update_func)