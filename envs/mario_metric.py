import torch
from torch import nn

from .utils import load_data_float


class MarioDistance(nn.Module):
    def __init__(self):
        super(MarioDistance, self).__init__()

        self.net = nn.Sequential(
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.LazyLinear(128),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1))


if __name__ == '__main__':
    from tqdm import trange
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_tensors, test_tensors = load_data_float()

    data_loader = torch.utils.data.DataLoader(training_tensors, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_tensors, batch_size=batch_size, shuffle=True)

    metric = MarioDistance().to(device)

    optim = torch.optim.AdamW(metric.parameters(), lr=1e-3)

    epochs = trange(1000)
    losses = []
    test_losses = []
    for epoch in epochs:
        losses.append(0)
        test_losses.append(0)
        for i, x in enumerate(data_loader):
            x_noise = x + torch.normal(0, torch.ones_like(x)*1e-1)

            idx = torch.randint(0, 2*x.shape[0], (2*x.shape[0],)).to(device)
            labels = torch.zeros_like(idx)
            labels[:x.shape[0]] = (idx[:x.shape[0]] < x.shape[0])*1
            labels[x.shape[0]:] = (idx[x.shape[0]:] >= x.shape[0])*1

            x = torch.concat((x, x_noise), dim=0)
            vectors = metric(x)
            distances = torch.linalg.vector_norm(vectors-vectors[idx], 2, dim=1)
            loss = torch.mean(labels*(distances**2)) + torch.mean((1-labels)*(torch.clip(1-distances, 0, None)**2))

            optim.zero_grad()
            loss.backward()
            optim.step()

            losses[-1] = (losses[-1]*i + loss.item())/(i+1)

            epochs.set_description(f'Losses: {losses[-1]} - {test_losses[-1]}')

        with torch.no_grad():
            for i, x in enumerate(test_loader):
                x_noise = x + torch.normal(0, torch.ones_like(x) * 1e-1)

                idx = torch.randint(0, 2 * x.shape[0], (2 * x.shape[0],)).to(device)


                x = torch.concat((x, x), dim=0)
                vectors = metric(x)
                distances = torch.linalg.vector_norm(vectors - vectors[idx], 2, dim=1)
                loss = torch.mean((distances ** 2))

                test_losses[-1] = (test_losses[-1] * i + loss.item()) / (i + 1)

                epochs.set_description(f'Losses: {losses[-1]} - {test_losses[-1]}')

    torch.save(metric, 'assets/checkpoints/metric.pt')

