import torch
from torch_geometric.nn.models import GAT
from torch_geometric.data import Data
from drawer import PygApp
from simulator import SunEarthSystem, make_dataset
from train import train

args = {
    'input_dim': 5,
    'output_dim': 2,
    'num_layers': 3,
    'hidden_dim': 16,
    'dropout': 0.05,
    'act': 'prelu',
    'norm': 'BatchNorm',
    'no_epochs': 100,
    'batch_size': 100,
    'lr': 0.01,
}

# dataset description:
# 1st dimension -> timesteps
# 2nd dimension -> bodies
# 3rd dimension -> body features
# features are: [mass(1), pos(2), vel(2), acc(2)]

dt = 0.1
system = SunEarthSystem()
dataset = make_dataset(system, relative=True, dt=dt, no_vel=1, no_steps=3000)
T = dataset.shape[0]   # no_timesteps

# creating adj list
adj = torch.eye(2, dtype=int).expand(T, -1, -1)

# converting dataset into pytorch friendly format
data_list = []
for t in range(T):
    x = torch.Tensor(dataset[t, :, :-2])
    y = torch.Tensor(dataset[t, :, -2:])
    data_list.append(Data(x=x, y=y, edge_index=adj[t]))

# initializing model
model = GAT(in_channels=args['input_dim'], 
            hidden_channels=args['hidden_dim'], 
            num_layers=args['num_layers'],
            out_channels=args['output_dim'], 
            dropout=args['dropout'],
            act=args['act'],
            norm=args['norm'])

# model training
train(model, data_list, args['no_epochs'], args['batch_size'], args['lr'])

# visualization
earth_mass = torch.Tensor(dataset[0, 0, 0:1])
earth_pos = torch.Tensor(dataset[0, 0, 1:3])
earth_pos_prev = torch.zeros(2)
earth_vel = torch.Tensor(dataset[0, 0, 3:-2])
earth_vel_prev = torch.zeros(2)

sun_mass = torch.Tensor(dataset[0, 1, 0:1])
sun_pos = torch.Tensor(dataset[0, 1, 1:3])
sun_pos_prev = torch.zeros(2)
sun_vel = torch.Tensor(dataset[0, 1, 3:-2])
sun_vel_prev = torch.zeros(2)

earth_sun_adj = torch.eye(2, dtype=int)

@torch.no_grad()
def update_func():
    global earth_pos, earth_pos_prev, earth_vel, earth_vel_prev
    global sun_pos, sun_pos_prev, sun_vel, sun_vel_prev
    
    points = [list(earth_pos) + [0], list(sun_pos) + [0]]
    
    earth_pos_delta = earth_pos - earth_pos_prev 
    earth_vel_delta = earth_vel - earth_vel_prev
    earth_x = torch.cat((earth_mass, earth_pos_delta, earth_vel_delta))

    sun_pos_delta = sun_pos - sun_pos_prev 
    sun_vel_delta = sun_vel - sun_vel_prev
    sun_x = torch.cat((sun_mass, sun_pos_delta, sun_vel_delta))

    x = torch.vstack((earth_x, sun_x))

    acc =  model(x, earth_sun_adj)

    earth_pos_prev = torch.clone(earth_pos)
    earth_vel_prev = torch.clone(earth_vel)
    earth_vel += acc[0, :] * dt
    earth_pos += earth_vel * dt

    sun_pos_prev = torch.clone(sun_pos)
    sun_vel_prev = torch.clone(sun_vel)
    sun_vel += acc[1, :] * dt
    sun_pos += sun_vel * dt  

    return points

app = PygApp(update_func)