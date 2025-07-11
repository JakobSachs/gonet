from state import Gamestate, Stone, Board

import numpy as np
from tinygrad.device import Device
from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit
import tinygrad.nn as nn
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, get_parameters
from tinygrad.nn.optim import Adam
import os

DEBUG = os.environ.get("DEBUG") == "1"


class Neuralnet:
    def __init__(self):
        # input: (9x9x3) (board_dim x board_dim x (black,white,valid_moves)
        # i think doing this should be fine, we dont need to pass in previous board since the ko-related info
        # is contained in the valid_moves channel, and we can ass state.blacks_points later to the model output
        # value-output is a single float giving the estimated value for black
        # policy-output is giving a ranking of best next moves for
        self.l1 = nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1)  # out: (9x9x32)
        self.l2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)  # out: (9x9x64)
        self.l3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)  # out: (9x9x128)
        self.l4 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)  # out: (9x9x256)
        self.l5 = nn.Linear(9 * 9 * 256, 1000)
        self.value_out = nn.Linear(1000, 1)
        self.policy_out = nn.Linear(1000, 9 * 9 + 1)  # 9x9 + 1 for pass

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.l1(x).relu()
        x = self.l2(x).relu()
        x = self.l3(x).relu()
        x = self.l4(x).relu().flatten(1)  # 9 * 9 * 256
        x = self.l5(x).relu()  # 1000
        value = self.value_out(x).tanh()
        policy = self.policy_out(x).sigmoid()
        return value, policy

    @TinyJit
    def predict(self, state: Gamestate) -> tuple[float, np.ndarray]:
        t = self._state_to_tens(state)
        v_out, p_out = self(t)
        return (float(v_out.numpy()[0, 0]), p_out.numpy()[0])

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

    def train_step(
        self, optim: Adam, x: Tensor, y_policy: Tensor, y_value: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        with Tensor.train():
            optim.zero_grad()
            v_out, p_out = self(x)

            # Loss calculation
            # Value loss: Mean Squared Error
            value_loss = (v_out - y_value).square().mean()

            # Policy loss: Cross-Entropy
            # p_out is sigmoid, y_policy is a probability distribution
            # We can use binary cross entropy
            policy_loss = -(
                y_policy * p_out.log() + (1 - y_policy) * (1 - p_out).log()
            ).mean()

            total_loss = (value_loss + policy_loss).backward()
            optim.step()
            return total_loss.realize(), value_loss.realize(), policy_loss.realize()

    def train(
        self,
        states: list[Gamestate],
        policy_targets: np.ndarray,
        value_targets: np.ndarray,
        epochs=10,
        batch_size=32,
        lr=0.001,
    ):
        optim = Adam(get_parameters(self), lr=lr)

        for epoch in range(epochs):
            epoch_loss, epoch_v_loss, epoch_p_loss = 0.0, 0.0, 0.0
            num_batches = (len(states) + batch_size - 1) // batch_size

            # Create a random permutation of indices
            indices = np.random.permutation(len(states))

            for i in range(num_batches):
                batch_indices = indices[i * batch_size : (i + 1) * batch_size]

                # Prepare batch data
                batch_states = [states[j] for j in batch_indices]
                X_batch = Tensor(
                    np.concatenate(
                        [self._state_to_tens(s).numpy() for s in batch_states], axis=0
                    )
                )
                Y_policy_batch = Tensor(policy_targets[batch_indices])
                Y_value_batch = Tensor(value_targets[batch_indices].reshape(-1, 1))

                loss, v_loss, p_loss = self.train_step(
                    optim, X_batch, Y_policy_batch, Y_value_batch
                )

                epoch_loss += loss.numpy()
                epoch_v_loss += v_loss.numpy()
                epoch_p_loss += p_loss.numpy()

            print(
                f"Epoch {epoch+1}/{epochs}, "
                f"Avg Loss: {epoch_loss/num_batches:.4f}, "
                f"Avg Value Loss: {epoch_v_loss/num_batches:.4f}, "
                f"Avg Policy Loss: {epoch_p_loss/num_batches:.4f}"
            )

    def save(self, path):
        std = get_state_dict(self)
        safe_save(std, path)


if __name__ == "__main__":
    from mcts import self_play_game

    net = Neuralnet()
    NUM_GAMES_PER_ITERATION = 10
    # Main training loop
    for i in range(10):  # 10 training iterations
        print(f"\n{'='*20} TRAINING ITERATION {i+1}/{10} {'='*20}")

        # 1. Generate training data from multiple self-play games
        print(
            f"\n--- Generating training data from {NUM_GAMES_PER_ITERATION} games ---"
        )
        all_states, all_policy_targets, all_value_targets = [], [], []

        for g in range(NUM_GAMES_PER_ITERATION):
            print(f"--- Playing game {g+1}/{NUM_GAMES_PER_ITERATION} ---")
            s = Gamestate.empty()

            states, policy_targets, value = self_play_game(
                net, num_simulations=250
            )  # Low sims for speed

            all_states.extend(states)
            all_policy_targets.extend(policy_targets)
            all_value_targets.extend(np.full(len(states), value, dtype=np.float32))
            print(
                f"Game finished. Winner: {'Black' if value > 0 else 'White' if value < 0 else 'Draw'}. Generated {len(states)} examples."
            )

        policy_targets_np = np.array(all_policy_targets, dtype=np.float32)
        value_targets_np = np.array(all_value_targets, dtype=np.float32)
        print(
            f"\nGenerated a total of {len(all_states)} examples from {NUM_GAMES_PER_ITERATION} games."
        )

        # 2. Train the network on the generated data
        print("\n--- Training the network ---")
        net.train(
            all_states, policy_targets_np, value_targets_np, epochs=5, batch_size=16
        )

        net.save("trained_model.pth")
