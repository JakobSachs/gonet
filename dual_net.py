"""
Dual-network Go bot placeholder.
- PolicyNet: estimates interesting moves (policy)
- ValueNet: estimates board value (value)
- DualNetGoBot: combines both for RL training
"""

import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from game import Game  # Fix for linter: ensure Game is defined for MCTSNode/MCTS
import time
import os

class PolicyNet:
    """
    Neural net to estimate move probabilities (policy).
    Input: board state as np.ndarray of shape (board_size, board_size)
    Output: move probabilities as np.ndarray of shape (board_size, board_size+1) (last entry is pass)
    """
    def __init__(self, board_size: int):
        self.board_size = board_size
        # Output: board_size*board_size + 1 (last is pass)
        self.weights = Tensor.uniform(board_size * board_size, board_size * board_size + 1)
        self.bias = Tensor.zeros(board_size * board_size + 1)

    def predict(self, board: np.ndarray) -> np.ndarray:
        """Return move probabilities for the given board state, last entry is pass."""
        x = Tensor(board.flatten().astype(np.float32))
        logits = x @ self.weights + self.bias
        probs = logits.softmax(axis=0).numpy()  # shape: (board_size*board_size+1,)
        # Print pass probability if DEBUG=1
        if os.environ.get("DEBUG") == "1":
            print(f"[PolicyNet] Pass probability: {probs[-1]:.4f}")
        # Return as (board_size, board_size) for board, and last entry for pass
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
    one_hot = np.zeros((batch_size, board_size * board_size + 1), dtype=np.float32)
    for i, (x, y) in enumerate(actions):
        if (x, y) == (-1, -1):
            one_hot[i, board_size * board_size] = 1.0  # last index is pass
        else:
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
    logits = state_tensor @ bot.policy_net.weights + bot.policy_net.bias  # (batch, board_size*board_size+1)
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

class MCTSNode:
    def __init__(self, game: 'Game', parent=None, move=None, prior=0.0):
        self.game = game  # Game object (deepcopy for each node)
        self.parent = parent
        self.move = move  # The move that led to this node
        self.children = {}  # move -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def is_expanded(self):
        return len(self.children) > 0

class MCTS:
    def __init__(self, bot, root_game: 'Game', num_simulations=50):
        import copy
        self.bot = bot
        self.root = MCTSNode(copy.deepcopy(root_game))
        self.num_simulations = num_simulations

    def run(self):
        for sim in range(self.num_simulations):
            node, path = self.root, [self.root]
            # Selection
            while node.is_expanded() and not node.game.game_over:
                node = self.select_child(node)
                path.append(node)
            # Expansion
            if not node.game.game_over:
                self.expand(node)
                if node.children:
                    node = self.select_child(node)
                    path.append(node)
            # Evaluation
            value = self.bot.value_net.predict(node.game.board.board)
            # Backpropagation
            for n in path:
                n.visit_count += 1
                n.value_sum += value
        # Choose move with highest visit count
        if not self.root.children:
            return None  # No moves
        best_move = max(self.root.children.items(), key=lambda item: item[1].visit_count)[0]
        return best_move

    def select_child(self, node):
        # UCB formula: value + exploration term
        C = 1.0
        total_visits = sum(child.visit_count for child in node.children.values())
        def ucb(child):
            return child.value() + C * child.prior * (np.sqrt(total_visits) / (1 + child.visit_count))
        return max(node.children.values(), key=ucb)

    def expand(self, node):
        # Use policy net to get priors for all legal moves (including pass)
        board = node.game.board.board
        priors = self.bot.policy_net.predict(board)  # shape: (board_size*board_size+1,)
        legal_moves = node.game.legal_moves()
        legal_moves_with_pass = legal_moves + [(-1, -1)]  # Always allow pass
        import copy
        for move in legal_moves_with_pass:
            if move == (-1, -1):
                prior = priors[-1]
                child_game = copy.deepcopy(node.game)
                child_game.pass_turn()
            else:
                x, y = move
                prior = priors[y * self.bot.board_size + x]
                child_game = copy.deepcopy(node.game)
                child_game.place_stone(x, y)
            node.children[move] = MCTSNode(child_game, parent=node, move=move, prior=prior)

class DualNetGoBot:
    """Combines PolicyNet and ValueNet for RL training."""
    def __init__(self, board_size: int = 9):
        self.policy_net = PolicyNet(board_size)
        self.value_net = ValueNet(board_size)
        self.board_size = board_size

    def select_move(self, board: np.ndarray, current_player=None) -> tuple:
        """Use MCTS to select a move."""
        import copy
        game = Game()
        game.board.board = copy.deepcopy(board)
        if current_player is not None:
            game.current_player = current_player
        mcts = MCTS(self, game, num_simulations=15)
        move = mcts.run()
        if move is None:
            # fallback: pick random legal move or pass
            legal_moves = game.legal_moves()
            legal_moves_with_pass = legal_moves + [(-1, -1)]
            move = legal_moves_with_pass[np.random.randint(len(legal_moves_with_pass))]
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
            epoch_total_loss = 0.0
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            batch_count = 0
            for i in range(0, len(experience), batch_size):
                batch_data = experience[i:i+batch_size]
                if len(batch_data) < batch_size:
                    continue
                states, actions, players, value_targets = zip(*batch_data)
                loss, policy_loss, value_loss = train_step(self, (states, actions, value_targets), optimizer)
                print(f"Batch {i//batch_size+1:3d} | Total Loss: {loss:.4f} | Policy Loss: {policy_loss:.4f} | Value Loss: {value_loss:.4f}")
                epoch_total_loss += loss
                epoch_policy_loss += policy_loss
                epoch_value_loss += value_loss
                batch_count += 1
            if batch_count > 0:
                avg_total_loss = epoch_total_loss / batch_count
                avg_policy_loss = epoch_policy_loss / batch_count
                avg_value_loss = epoch_value_loss / batch_count
                print(f"End of epoch {epoch+1} | Avg Total Loss: {avg_total_loss:.4f} | Avg Policy Loss: {avg_policy_loss:.4f} | Avg Value Loss: {avg_value_loss:.4f}\n")
            else:
                print(f"End of epoch {epoch+1} | No full batches processed.\n")
        Tensor.training = False  # Reset after training
        if epochs > 0 and batch_count > 0:
            print(f"Training complete. Final Avg Total Loss: {avg_total_loss:.4f} | Avg Policy Loss: {avg_policy_loss:.4f} | Avg Value Loss: {avg_value_loss:.4f}")

def self_play_game(bot, board_size=9):
    """
    Play a self-play game using the bot. Collects (state, action, player) for each move.
    At the end, assigns the final result (+1/-1) as the value target for each state.
    Returns: list of (state, action, player, value_target)
    """
    from board import Board
    import time
    states = []
    actions = []
    players = []
    board = Board()
    current_player = board.BLACK
    game_over = False
    passes = 0
    move_num = 1
    MAX_MOVES = 200
    print("\n[Self-Play] Starting a new self-play game...")
    # Use Game object for pass logic
    from game import Game
    game = Game()
    start_time = time.time()
    while not game.game_over and move_num <= MAX_MOVES:
        state = game.board.board.copy()
        legal_moves = game.legal_moves()
        legal_moves_with_pass = legal_moves + [(-1, -1)]
        if not legal_moves:
            move = (-1, -1)  # must pass
            print(f"[Self-Play] No legal moves. Forced pass.")
        else:
            if move_num == 1 or move_num % 10 == 0:
                elapsed = time.time() - start_time
                mps = move_num / elapsed if elapsed > 0 else 0.0
                print(f"[Self-Play] Move {move_num}, Player: {'Black' if game.current_player == board.BLACK else 'White'} | {move_num} moves, {mps:.2f} moves/sec")
            move = bot.select_move(game.board.board, current_player=game.current_player)
            if move not in legal_moves_with_pass:
                if move_num == 1 or move_num % 10 == 0:
                    print(f"[Self-Play] Bot picked illegal move {move}, choosing random legal move or pass.")
                move = legal_moves_with_pass[np.random.randint(len(legal_moves_with_pass))]
        if move == (-1, -1):
            if legal_moves:
                print(f"[Self-Play] Bot chose to pass (not forced). Player {'Black' if game.current_player == board.BLACK else 'White'}.")
            else:
                print(f"[Self-Play] Player {'Black' if game.current_player == board.BLACK else 'White'} passes (forced).")
            game.pass_turn()
            print(f"[Self-Play] Called game.pass_turn(). Passes: {game.passes}, Game over: {game.game_over}")
        else:
            x, y = move
            game.place_stone(x, y)
        states.append(state)
        actions.append(move)
        players.append(game.current_player)
        move_num += 1
    if move_num > MAX_MOVES:
        print(f"[Self-Play] Max moves ({MAX_MOVES}) reached, terminating game.")
        game.end_game()
    elapsed = time.time() - start_time
    mps = (move_num-1) / elapsed if elapsed > 0 else 0.0
    print(f"[Self-Play] Game over. {move_num-1} moves played in {elapsed:.2f} seconds. {mps:.2f} moves/sec.")
    black_score = np.sum(game.board.board == board.BLACK)
    white_score = np.sum(game.board.board == board.WHITE)
    print(f"[Self-Play] Game over. Black: {black_score}, White: {white_score}")
    if black_score > white_score:
        result = board.BLACK
        print("[Self-Play] Black wins!")
    elif white_score > black_score:
        result = board.WHITE
        print("[Self-Play] White wins!")
    else:
        result = 0  # Draw
        print("[Self-Play] Draw!")
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
    num_rl_iterations = 10  # Number of RL loops
    num_games = 5           # Self-play games per iteration
    for rl_iter in range(1, num_rl_iterations + 1):
        print(f"\n=== RL Iteration {rl_iter}/{num_rl_iterations} ===")
        dataset = []
        for _ in range(num_games):
            game = self_play_game(bot, 9)
            for (state, action, player, value_target) in game:
                dataset.append((state.astype(np.float32), action, player, np.float32(value_target)))
        print(f"[RL] Training on {len(dataset)} samples from {num_games} self-play games...")
        bot.train(dataset, batch_size=16, lr=1e-3, epochs=20)
    print("\n[RL] Training complete.")

