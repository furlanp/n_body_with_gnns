import random
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from simulator import SunEarthSystem
from InteractionNetwork import InteractionNetwork
from drawer import PygApp
from simulator import make_dataset
import copy

torch.manual_seed(2)
random.seed(5)

device = None
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

dt = 0.1
system = SunEarthSystem()
dataset = make_dataset(system, relative=False, dt=dt, no_vel=1, no_steps=3000)
T = dataset.shape[0]   # no_timesteps

no_objects = 2
no_relations = 2
relation_dim = 1
effect_dim = 100
object_dim = 5

def get_batch(data, batch_size):
    # rand_idx  = [random.randint(0, len(data) - 2) for i in range(batch_size)]
    rand_idx  = random.sample(range(len(data) - 2), batch_size)
    label_idx = [idx + 1 for idx in rand_idx]
    
    batch_data = data[rand_idx]
    label_data = data[label_idx]
        
    # receiver_relations and sender_relations are one-hot encoding matrices
    # each column indicates the receiver and sender object index
    receiver_relations = np.zeros((batch_size, no_objects, no_relations), dtype=float)
    sender_relations = np.zeros((batch_size, no_objects, no_relations), dtype=float)
    
    cnt = 0
    for i in range(no_objects):
        for j in range(no_objects):
            if i != j:
                receiver_relations[:, i, cnt] = 1.0
                sender_relations[:, j, cnt] = 1.0
                cnt += 1
    
    #There is no relation info in solar system task, just fill with zeros
    relation_info = np.zeros((batch_size, no_relations, relation_dim))
    target = label_data[:, :, 3:]
    
    objects = Variable(torch.FloatTensor(batch_data).to(device))
    sender_relations = Variable(torch.FloatTensor(sender_relations).to(device))
    receiver_relations = Variable(torch.FloatTensor(receiver_relations).to(device))
    relation_info = Variable(torch.FloatTensor(relation_info).to(device))
    target = Variable(torch.FloatTensor(target).to(device)).view(-1, 2)

    return objects, sender_relations, receiver_relations, relation_info, target


model = InteractionNetwork(no_objects, object_dim, no_relations, relation_dim, effect_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

no_epoch = 650
batches_per_epoch = 1
losses = []

# best_model = None
# best_loss = 100000

# model training loop
for epoch in range(no_epoch):
    for i in range(batches_per_epoch):
        objects, sender_relations, receiver_relations, relation_info, target = get_batch(dataset[:, :, :-2], T - 2)
        predicted = model(objects, sender_relations, receiver_relations, relation_info)
        loss = loss_fn(predicted, target)
        # if loss < best_loss:
        #     best_model = copy.deepcopy(model)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(np.sqrt(loss.item()))
        print(f'epoch: {epoch}, batch_idx: {i}, loss: {losses[-1]}')

# visualization
objects = torch.FloatTensor(dataset[0, :, :-2]).unsqueeze(0).to(device)
receiver_relations = torch.eye(no_objects).unsqueeze(0).to(device)
sender_relations = (torch.ones((no_objects, no_objects)) - torch.eye(no_objects)).unsqueeze(0).to(device)
relation_info = torch.ones((1, no_relations, relation_dim)).to(device)
print(objects.shape, receiver_relations.shape, sender_relations.shape, relation_info.shape)

@torch.no_grad()
def update_func():
    global objects
    predicted = model(objects, sender_relations, receiver_relations, relation_info)
    objects[0, :, 3:] = predicted   # new speed
    objects[0, :, 1:3] += objects[0, :, 3:] * 0.1   # new position
    return [list(objects[0, 0, 1:3]) + [0], list(objects[0, 1, 1:3]) + [0]]

app = PygApp(update_func)