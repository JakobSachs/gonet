from state import Gamestate, Stone, Board

import numpy as np
from tinygrad.device import Device
from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit
import tinygrad.nn as nn
from tinygrad.nn.state import safe_save, safe_load, get_state_dict


class MLPBlock:
    def __init__(self, in_dim, hidden_dim, out_dim):
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None
    def __call__(self, x):
        out = self.fc1(x).relu()  # type: ignore[attr-defined]
        out = self.fc2(out)
        if self.proj is not None:
            x_proj = self.proj(x)
        else:
            x_proj = x
        return (out + x_proj).relu()  # type: ignore[attr-defined]


class Valuenet:
    def __init__(self, num_blocks=3, input_shape=(9, 9, 3), hidden_dim=256):
        self.input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.input_layer = nn.Linear(self.input_dim, hidden_dim)
        self.blocks = [MLPBlock(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_blocks)]
        self.out = nn.Linear(hidden_dim, 1)

    def __call__(self, x: Tensor) -> Tensor:
        x = x.reshape(x.shape[0], -1)  # type: ignore[attr-defined]
        x = self.input_layer(x).relu()  # type: ignore[attr-defined]
        for block in self.blocks:
            x = block(x)
        return self.out(x.tanh())  # type: ignore[attr-defined]

    @TinyJit
    def predict(self, state: Gamestate) -> float:
        t = self._state_to_tens(state)
        arr = self(t).numpy()  # type: ignore[attr-defined]
        return float(arr.flatten()[0])  # type: ignore[arg-type]

    @staticmethod
    def _state_to_tens(state: Gamestate) -> Tensor:
        board = state.board
        blacks = np.where(board == Stone.BLACK.value, board, 0).astype(np.float32)
        whites = np.where(board == Stone.WHITE.value, board, 0).astype(np.float32)
        moves = np.zeros((9, 9), dtype=np.float32)
        for m in state.get_moves():
            moves[m] = 1.0
        tens = np.stack((blacks, whites, moves), axis=-1)
        return Tensor(np.expand_dims(tens, axis=0))

    @staticmethod
    def _augment(tensor: Tensor):
        return [
            tensor,
            tensor.flip(1),
            tensor.flip(2),
            tensor.flip(1).flip(2),
        ]

    @TinyJit
    def train_step(self, optim, X: Tensor, Y: Tensor):
        print(optim, X, Y)
        optim.zero_grad()
        out = self(X)
        diff = out - Y
        loss = (diff * diff).mean()  # type: ignore[attr-defined]
        loss.backward()  # type: ignore[attr-defined]
        optim.step()
        return loss

    def train(
        self,
        states: list[Gamestate],
        outcomes: list[float],
        epochs=100,
        batch_size=128,
        lr=0.001,
    ):
        tensors = [self._state_to_tens(s) for s in states]
        augmented_tensors = []
        augmented_outcomes = []
        for t, o in zip(tensors, outcomes):
            augmented_t = self._augment(t)
            augmented_tensors.extend(augmented_t)
            augmented_outcomes.extend([o] * len(augmented_t))
        perm = np.random.permutation(len(augmented_tensors))
        augmented_tensors = [augmented_tensors[i] for i in perm]
        augmented_outcomes = [augmented_outcomes[i] for i in perm]
        from tinygrad.nn.optim import Adam
        from tinygrad.nn.state import get_parameters
        optim = Adam(get_parameters(self), lr=lr)
        with Tensor.train():
            for epoch in range(epochs):
                for i in range(0, len(augmented_tensors), batch_size):
                    batch_X_list = augmented_tensors[i : i + batch_size]
                    batch_Y_list = augmented_outcomes[i : i + batch_size]
                    if not batch_X_list:
                        continue
                    if len(batch_X_list) < batch_size:
                        continue
                    X = Tensor.cat(*batch_X_list, dim=0)
                    Y = Tensor(batch_Y_list, requires_grad=False).reshape(-1, 1)
                    loss = self.train_step(optim, X, Y)
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}, loss {float(loss):.4f}")

    def save(self, path):
        from tinygrad.nn.state import get_state_dict, safe_save
        std = get_state_dict(self)
        safe_save(std, path)


if __name__ == "__main__":
    s = Gamestate.empty()
    net = Valuenet()
    pred_before = net.predict(s)
    if hasattr(pred_before, 'numpy'):
        pred_before = float(pred_before.numpy().flatten()[0])  # type: ignore[attr-defined, arg-type]
    else:
        pred_before = float(pred_before)  # type: ignore[arg-type]
    print(f"prediction before training: {pred_before:.4f}")

    # create some dummy training data
    states = [Gamestate.empty() for _ in range(20)]
    outcomes = list(np.random.rand(len(states)))

    net.train(states, outcomes, epochs=100)

    pred_after = net.predict(s)
    if hasattr(pred_after, 'numpy'):
        pred_after = float(pred_after.numpy().flatten()[0])  # type: ignore[attr-defined, arg-type]
    else:
        pred_after = float(pred_after)  # type: ignore[arg-type]
    print(f"prediction after training: {pred_after:.4f}")
