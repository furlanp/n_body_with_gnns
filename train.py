import torch
from torch_geometric.loader import DataLoader
from random import sample

# function that trains the model by using the given optimizer and loss_fn
def train_batch(model, batch, optimizer, loss_fn):
    model.train()

    optimizer.zero_grad()
    out = model(batch.x, batch.edge_index)
    loss = loss_fn(out.reshape(-1), batch.y.reshape(-1))

    loss.backward()
    optimizer.step()

    return loss.item()
 
def train(model, data_list, no_epochs, batch_size=64, lr=0.01):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    print('training started...')

    for epoch in range(no_epochs):
        loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)
        for batch_idx, batch in enumerate(loader):
            loss = train_batch(model, batch, opt, loss_fn)
            print(f'epoch: {epoch}, batch_idx: {batch_idx}, loss: {loss}')

    print('training finished...')