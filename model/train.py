import random
from collections import deque
import torch

def trainer(
    model,
    train_queue,
    cfg
):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    buffer = deque(maxlen=cfg.buf_size)

    # warm up
    while len(buffer) < cfg.warmup_size:
        buffer.append(train_queue.get())

    while True:
        # fill buffer a bit
        for _ in range(cfg.fill_steps):
            buffer.append(train_queue.get())
        # do some gradient steps
        for _ in range(cfg.train_steps):
            batch = random.sample(buffer, cfg.batch_size)
            states = torch.stack([b[0] for b in batch])
            pis = torch.tensor([b[1] for b in batch], dtype=torch.float32)
            zs = torch.tensor([b[2] for b in batch], dtype=torch.float32)
            # forward
            vs = model(states).squeeze()
            loss = F.mse_loss(vs, zs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
