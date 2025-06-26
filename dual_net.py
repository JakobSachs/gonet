"""
Dual-network Go bot placeholder.
- PolicyNet: estimates interesting moves (policy)
- ValueNet: estimates board value (value)
- DualNetGoBot: combines both for RL training
"""

import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam

class PolicyNet:
    """
    Neural net to estimate move probabilities (policy).
    Input: board state as np.ndarray of shape (board_size, board_size)
    Output: move probabilities as np.ndarray of shape (board_size, board_size)
    """
    def __init__(self, board_size: int):
        self.board_size = board_size
        # Simple linear model: flatten board -> logits for each position
        self.weights = Tensor.uniform(board_size * board_size, board_size * board_size)
        self.bias = Tensor.zeros(board_size * board_size)

    def predict(self, board: np.ndarray) -> np.ndarray:
        """Return move probabilities for the given board state."""
        x = Tensor(board.flatten().astype(np.float32))
        logits = x @ self.weights + self.bias
        probs = logits.softmax(axis=0).numpy().reshape((self.board_size, self.board_size))
        return probs

class ValueNet:
    """
    Neural net to estimate the value of a board state.
    Input: board state as np.ndarray of shape (board_size, board_size)
    Output: scalar value (float)
    """
    def __init__(self, board_size: int):
        self.board_size = board_size
        # Simple linear model: flatten board -> scalar value
        self.weights = Tensor.uniform(board_size * board_size, 1)
        self.bias = Tensor.zeros(1)

    def predict(self, board: np.ndarray) -> float:
        """Return value estimate for the given board state."""
        x = Tensor(board.flatten().astype(np.float32))
        value = (x @ self.weights + self.bias).tanh().numpy().item()
        return value

# --- Loss and training utilities ---
def one_hot_action(actions, board_size):
    batch_size = len(actions)
    one_hot = np.zeros((batch_size, board_size * board_size), dtype=np.float32)
    for i, (x, y) in enumerate(actions):
        one_hot[i, y * board_size + x] = 1.0
    return one_hot

def train_step(bot, batch, optimizer):
    states, actions, value_targets = batch
    board_size = bot.board_size
    batch_size = len(states)
    # Prepare tensors
    state_tensor = Tensor(np.stack(states).reshape(batch_size, -1).astype(np.float32))  # (batch, board_size*board_size)
    action_tensor = Tensor(one_hot_action(actions, board_size).astype(np.float32))
    value_tensor = Tensor(np.array(value_targets, dtype=np.float32).reshape(-1, 1))
    # Policy forward
    logits = state_tensor @ bot.policy_net.weights + bot.policy_net.bias  # (batch, board_size*board_size)
    policy_probs = logits.softmax(axis=1)
    # Policy loss (cross-entropy)
    policy_loss = -(action_tensor * policy_probs.log()).sum() / batch_size
    # Value forward
    value_pred = (state_tensor @ bot.value_net.weights + bot.value_net.bias).tanh()
    # Value loss (MSE)
    value_loss = ((value_pred - value_tensor) ** 2).mean()
    # Total loss
    loss = policy_loss + value_loss
    # Backprop and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.numpy(), policy_loss.numpy(), value_loss.numpy()

class DualNetGoBot:
    """Combines PolicyNet and ValueNet for RL training."""
    def __init__(self, board_size: int = 9):
        self.policy_net = PolicyNet(board_size)
        self.value_net = ValueNet(board_size)
        self.board_size = board_size

    def select_move(self, board: np.ndarray) -> tuple:
        """Use policy net to select a move (placeholder: random)."""
        probs = self.policy_net.predict(board)
        move = np.unravel_index(np.argmax(probs), probs.shape)
        return move

    def evaluate(self, board: np.ndarray) -> float:
        """Use value net to estimate board value."""
        return self.value_net.predict(board)

    def train(self, experience, batch_size=16, lr=1e-3, epochs=3):
        Tensor.training = True  # Enable training mode for tinygrad
        optimizer = Adam([
            self.policy_net.weights, self.policy_net.bias,
            self.value_net.weights, self.value_net.bias
        ], lr=lr)
        for epoch in range(epochs):
            np.random.shuffle(experience)
            print(f"\n=== Epoch {epoch+1}/{epochs} ===")
            for i in range(0, len(experience), batch_size):
                batch_data = experience[i:i+batch_size]
                if len(batch_data) < batch_size:
                    continue
                states, actions, players, value_targets = zip(*batch_data)
                loss, policy_loss, value_loss = train_step(self, (states, actions, value_targets), optimizer)
                print(f"Batch {i//batch_size+1:3d} | Total Loss: {loss:.4f} | Policy Loss: {policy_loss:.4f} | Value Loss: {value_loss:.4f}")
            print(f"End of epoch {epoch+1}\n")
        Tensor.training = False  # Reset after training

def self_play_game(bot, board_size=9):
    """
    Play a self-play game using the bot. Collects (state, action, player) for each move.
    At the end, assigns the final result (+1/-1) as the value target for each state.
    Returns: list of (state, action, player, value_target)
    """
    from board import Board
    states = []
    actions = []
    players = []
    board = Board()
    current_player = board.BLACK
    game_over = False
    passes = 0
    while not game_over:
        state = board.board.copy()
        # Mask illegal moves
        legal_moves = [(x, y) for y in range(board.SIZE) for x in range(board.SIZE) if board.board[y, x] == board.EMPTY]
        if not legal_moves or passes == 2:
            game_over = True
            break
        # Bot selects move
        move = bot.select_move(state)
        if move not in legal_moves:
            # If bot picks illegal move, pick random legal move
            move = legal_moves[np.random.randint(len(legal_moves))]
        x, y = move
        board.board[y, x] = current_player
        states.append(state)
        actions.append(move)
        players.append(current_player)
        # Alternate player
        current_player = board.BLACK if current_player == board.WHITE else board.WHITE
        passes = 0  # For simplicity, don't handle pass moves here
        # Check for end (very simple: end when board is full)
        if np.all(board.board != board.EMPTY):
            game_over = True
    # Assign value targets: +1 for winner, -1 for loser
    # For demo, just count stones
    black_score = np.sum(board.board == board.BLACK)
    white_score = np.sum(board.board == board.WHITE)
    if black_score > white_score:
        result = board.BLACK
    elif white_score > black_score:
        result = board.WHITE
    else:
        result = 0  # Draw
    value_targets = []
    for p in players:
        if result == 0:
            value_targets.append(0.0)
        elif p == result:
            value_targets.append(1.0)
        else:
            value_targets.append(-1.0)
    return list(zip(states, actions, players, value_targets))

if __name__ == "__main__":
    from board import Board
    bot = DualNetGoBot(9)
    print("\n--- RL Training with self-play data ---")
    dataset = []
    num_games = 20
    for _ in range(num_games):
        game = self_play_game(bot, 9)
        for (state, action, player, value_target) in game:
            dataset.append((state.astype(np.float32), action, player, np.float32(value_target)))
    bot.train(dataset, batch_size=16, lr=1e-3, epochs=10) 