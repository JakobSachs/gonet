from state import Gamestate, Stone, Board

import numpy as np
from tinygrad import Device, Tensor, TinyJit, nn
from tinygrad.nn.state import safe_save, safe_load, get_state_dict


class Valuenet:
    def __init__(self):
        # input: (9x9x3) (board_dim x board_dim x (black,white,valid_moves)
        # i think doing this should be fine, we dont need to pass in previous board since the ko-related info
        # is contained in the valid_moves channel, and we can ass state.blacks_points later to the model output
        # Output is a single float giving the estimated value for black
        self.l1 = nn.Conv2d(3, 32, kernel_size=(3, 3))  # out (7x7x16)
        self.l2 = nn.Conv2d(32, 64, kernel_size=(3, 3))  # out: (5x5x32)
        self.l3 = nn.Linear(5 * 5 * 64, 100)
        self.out = nn.Linear(100, 1)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.l1(x).relu()
        x = self.l2(x).relu()
        x = self.l3(x.flatten(1))
        return self.out(x.tanh())

    @TinyJit
    def predict(self, state: Gamestate) -> float:
        t = self._state_to_tens(state)
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
            tensor.flip(2),  # vertical flip
            tensor.flip(3),  # horizontal flip
            tensor.flip(2).flip(3),  # 180 degree rotation
        ]

    @TinyJit
    def train_step(self, optim, X: Tensor, Y: Tensor):
        print(optim, X, Y)
        optim.zero_grad()
        out = self(X)
        loss = (out - Y).square().mean()
        loss.backward()
        optim.step()
        return loss.realize()

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
                    print(f"Epoch {epoch+1}, loss {loss.item():.4f}")

    def save(self, path):
        std = get_state_dict(self)
        safe_save(std, path)


if __name__ == "__main__":
    s = Gamestate.empty()
    net = Valuenet()
    print(f"prediction before training: {net.predict(s):.4f}")

    # create some dummy training data
    states = [Gamestate.empty() for _ in range(20)]
    outcomes = list(np.random.rand(len(states)))

    net.train(states, outcomes, epochs=100)

    print(f"prediction after training: {net.predict(s):.4f}")
