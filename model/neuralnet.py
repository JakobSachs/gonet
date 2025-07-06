from state import Gamestate, Stone, Board

import numpy as np
from tinygrad.device import Device
from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit
import tinygrad.nn as nn
from tinygrad.nn.state import safe_save, safe_load, get_state_dict
import os

DEBUG = os.environ.get("DEBUG") == "1"


class Valuenet:
    def __init__(self):
        # input: (9x9x3) (board_dim x board_dim x (black,white,valid_moves)
        # i think doing this should be fine, we dont need to pass in previous board since the ko-related info
        # is contained in the valid_moves channel, and we can ass state.blacks_points later to the model output
        # Output is a single float giving the estimated value for black
        self.l1 = nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1)  # out: (9x9x32)
        self.l2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)  # out: (9x9x64)
        self.l3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)  # out: (9x9x128)
        self.l4 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)  # out: (9x9x256)
        self.l5 = nn.Linear(9 * 9 * 256, 100)
        self.out = nn.Linear(100, 1)

    def __call__(self, x: Tensor) -> Tensor:
        if DEBUG:
            print(f"[Valuenet] Input shape: {x.shape}")
        x = self.l1(x).relu()
        if DEBUG:
            print(f"[Valuenet] After l1: {x.shape}")
        x = self.l2(x).relu()
        if DEBUG:
            print(f"[Valuenet] After l2: {x.shape}")
        x = self.l3(x).relu() 
        if DEBUG:
            print(f"[Valuenet] After l3: {x.shape}")
        x = self.l4(x).relu().flatten(1)
        if DEBUG:
            print(f"[Valuenet] After l4: {x.shape}")
        x = self.l5(x).relu()
        if DEBUG:
            print(f"[Valuenet] After l5: {x.shape}")
        out = self.out(x.tanh())
        if DEBUG:
            print(f"[Valuenet] Output: {out}")
        return out

    @TinyJit
    def predict(self, state: Gamestate) -> float:
        t = self._state_to_tens(state)
        if DEBUG:
            print(f"[Valuenet.predict] State: {state}")
        return float(self(t).numpy()[0, 0])

    @staticmethod
    def _state_to_tens(state: Gamestate) -> Tensor:
        board = state.board
        blacks = np.where(board == Stone.BLACK.value, board, 0).astype(np.float32)
        whites = np.where(board == Stone.WHITE.value, board, 0).astype(np.float32)
        moves = np.zeros((9, 9), dtype=np.float32)
        for m in state.get_moves():
            moves[m] = 1.0
        return Tensor(np.expand_dims(np.stack((blacks, whites, moves), axis=0), axis=0))

    @staticmethod
    def _augment(tensor: Tensor):
        # The 8 symmetries of a square for a (N, C, H, W) tensor
        return [
            tensor,
            # flips
            tensor.flip(2),  # vertical flip
            tensor.flip(3),  # horizontal flip
            tensor.flip(2).flip(3),  # 180 degree rotation
            # rotations
            tensor[:, :, ::-1, :],  # 90 degree rotation
            tensor[:, :, :, ::-1],  # 180 degree rotation
            tensor[:, :, ::-1, ::-1],  # 270 degree rotation
        ]

    @TinyJit
    def train_step(self, optim, X: Tensor, Y: Tensor):
        print(optim, X, Y)
        optim.zero_grad()
        out = self(X)
        diff = out - Y
        loss = diff * diff
        loss = loss.sum() / loss.numel()  # type: ignore[attr-defined]
        loss.backward()  # type: ignore[attr-defined]
        optim.step()
        return loss.realize()  # type: ignore[attr-defined]

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
        augmented_tensors = [augmented_tensors[i].realize() for i in perm]
        augmented_outcomes = [augmented_outcomes[i] for i in perm]

        optim = nn.optim.Adam(nn.state.get_parameters(self), lr=lr)
        with Tensor.train():
            for epoch in range(epochs):
                for i in range(0, len(augmented_tensors), batch_size):
                    batch_X_list = augmented_tensors[i : i + batch_size]
                    batch_Y_list = augmented_outcomes[i : i + batch_size]

                    if not batch_X_list:
                        continue

                    # if the batch is smaller than the batch size, we skip it to avoid JIT errors
                    if len(batch_X_list) < batch_size:
                        continue

                    X = Tensor.cat(*batch_X_list, dim=0)
                    Y = Tensor(batch_Y_list, requires_grad=False).reshape(-1, 1)
                    loss = self.train_step(optim, X, Y)

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}, loss {float(loss.numpy()):.4f}")  # type: ignore[arg-type]

    def save(self, path):
        std = get_state_dict(self)
        safe_save(std, path)


if __name__ == "__main__":
    s = Gamestate.empty()
    net = Valuenet()
    print(f"prediction before training: {float(net.predict(s)):.4f}")  # type: ignore[arg-type]

    # create some dummy training data
    states = [Gamestate.empty() for _ in range(20)]
    outcomes = list(np.random.rand(len(states)))

    net.train(states, outcomes, epochs=100)

    print(f"prediction after training: {float(net.predict(s)):.4f}")  # type: ignore[arg-type]
