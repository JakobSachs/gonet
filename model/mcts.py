import math
import os
import numpy as np
from tqdm import tqdm

from state import Gamestate
from neuralnet import Neuralnet

DEBUG = os.environ.get("DEBUG") == "1"

class Node:
    """
    A node in the Monte Carlo Tree Search. Stores information about a single game state.
    """

    def __init__(
        self, state: Gamestate, parent: "Node | None" = None, prior: float = 0.0
    ):
        self.state = state
        self.parent = parent
        self.children: dict[tuple[int, int], Node] = {}  # move -> Node
        self.visit_count = 0
        self.total_value = 0.0  # From black's perspective
        self.prior = prior  # Prior probability of selecting this node

    @property
    def average_value(self) -> float:
        """
        The average value of this node, from black's perspective.
        """
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def is_fully_expanded(self) -> bool:
        """
        Checks if all possible moves from this node have been expanded.
        """
        return len(self.children) > 0 and len(self.children) == len(
            self.state.get_moves()
        )

    def select_child(self, c_puct: float) -> "Node":
        """
        Selects the best child node according to the PUCT formula.
        This formula balances exploration and exploitation.
        """
        best_score = -float("inf")
        best_child = None
        best_move = None

        # Get all children's scores first for debug printing
        if DEBUG:
            child_scores = {}
            for move, child in self.children.items():
                q_value = child.average_value
                if not self.state.blacks_turn:
                    q_value = -q_value
                u_value = (c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count))
                child_scores[move] = q_value + u_value
            
            # Sort by score and print top 5
            sorted_moves = sorted(child_scores.items(), key=lambda item: item[1], reverse=True)
            print("Top 5 child scores:")
            for move, score in sorted_moves[:5]:
                print(f"  Move: {move}, Score: {score:.3f}")


        for move, child in self.children.items():
            # Q-value (exploitation) + U-value (exploration)
            q_value = child.average_value
            # The value is always from black's perspective. If it's white's turn, we want to minimize black's score.
            if not self.state.blacks_turn:
                q_value = -q_value

            u_value = (
                c_puct
                * child.prior
                * math.sqrt(self.visit_count)
                / (1 + child.visit_count)
            )
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_child = child
                best_move = move

        if best_child is None:
            raise Exception("Could not select a child node")
        
        if DEBUG:
            print(f"Selected move: {best_move} with score {best_score:.3f}")

        return best_child

    def expand(self, policy: np.ndarray):
        """
        Expands this node by creating children for all valid moves.
        The policy from the neural network determines the prior probabilities.
        """
        valid_moves = self.state.get_moves()
        for move in valid_moves:
            if move not in self.children:
                # Map move to policy index
                if move == (-1, -1):  # pass move
                    idx = 81
                else:
                    x, y = move
                    idx = y * 9 + x

                prior_prob = policy[idx]
                new_state = self.state.do_move(move)
                self.children[move] = Node(new_state, parent=self, prior=prior_prob)

    def backpropagate(self, value: float):
        """
        Recursively updates the visit count and total value of this node and all its parents.
        """
        self.visit_count += 1
        self.total_value += value
        if self.parent:
            # The value is passed up unchanged (always from black's perspective)
            self.parent.backpropagate(value)


def run_mcts(
    state: Gamestate, nnet: Neuralnet, num_simulations: int, c_puct: float = 1.0, move_count: int = 0
) -> tuple[tuple[int, int], np.ndarray]:
    """
    Runs the MCTS algorithm to determine the best move from the current state.
    Returns the best move and the policy target (normalized visit counts).
    """
    root = Node(state)

    for i in range(num_simulations):
        if DEBUG:
            print(f"\n--- Simulation {i+1}/{num_simulations} ---")
        node = root
        # 1. Selection
        path = [node]
        while node.is_fully_expanded() and not node.state.game_over:
            node = node.select_child(c_puct)
            path.append(node)
        
        if DEBUG:
            print(f"Selection path length: {len(path)}")
            print(f"Selected node state:\n{node.state.board}")

        # 2. Expansion & 3. Evaluation
        if not node.state.game_over:
            # Use the neural network to get policy and value
            value, policy = nnet.predict(node.state)
            if DEBUG:
                print(f"NN value: {value:.4f}")
                valid_moves = node.state.get_moves()
                policy_probs = {move: policy[81] if move == (-1,-1) else policy[move[1]*9 + move[0]] for move in valid_moves}
                sorted_policy = sorted(policy_probs.items(), key=lambda item: item[1], reverse=True)
                print(f"Top 5 NN policy moves: { {k: f'{v:.3f}' for k, v in sorted_policy[:5]} }")

            node.expand(policy)
            # The value from the NN is from black's perspective
            node.backpropagate(value)
            if DEBUG:
                print(f"Backpropagated value {value:.4f} up the path.")
        else:
            # For terminal nodes, the value is the final score of the game
            score = node.state.score()
            value = 1.0 if score > 0 else -1.0 if score < 0 else 0.0
            if DEBUG:
                print(f"Terminal node reached. Final score: {score}, Value: {value}")
            node.backpropagate(value)

    # After simulations, choose the move with the highest visit count
    if not root.children:
        return (-1, -1), np.zeros(82, dtype=np.float32)

    # Create policy target
    policy_target = np.zeros(82, dtype=np.float32)
    for move, child in root.children.items():
        idx = 81 if move == (-1, -1) else move[1] * 9 + move[0]
        policy_target[idx] = child.visit_count
    policy_target /= np.sum(policy_target) # Normalize

    if DEBUG:
        print("\n--- MCTS Results (Top 5) ---")
        sorted_children = sorted(root.children.items(), key=lambda item: item[1].visit_count, reverse=True)
        for move, child_node in sorted_children[:5]:
            print(
                f"Move: {move}, Visits: {child_node.visit_count}, Avg Value: {child_node.average_value:.4f}"
            )

    children_to_consider = root.children
    if move_count < 50:
        if (-1, -1) in children_to_consider and len(children_to_consider) > 1:
            children_to_consider = {m: c for m, c in children_to_consider.items() if m != (-1, -1)}

    best_move = max(children_to_consider.items(), key=lambda item: item[1].visit_count)[0]
    return best_move, policy_target




def self_play_game(net: Neuralnet, num_simulations: int):
    """
    Plays a single game of self-play, returning the training data.
    Returns: (states, policy_targets, value)
    """
    game_state = Gamestate.empty()
    states, policy_targets = [], []
    
    move_count = 0
    max_moves = 200

    with tqdm(total=max_moves, desc="Self-play moves", unit="move") as pbar:
        while not game_state.game_over and move_count < max_moves:
            best_move, policy = run_mcts(game_state, net, num_simulations=num_simulations, move_count=move_count)
            
            states.append(game_state)
            policy_targets.append(policy)

            try:
                game_state = game_state.do_move(best_move)
                move_count += 1
                pbar.update(1)
            except ValueError as e:
                print(f"Invalid move chosen by MCTS: {best_move}. Error: {e}")
                break # End game on error

    final_score = game_state.score()
    value = 1.0 if final_score > 0 else -1.0 if final_score < 0 else 0.0
    
    return states, policy_targets, value


if __name__ == "__main__":
    # Example of generating training data from one self-play game
    net = Neuralnet()
    
    print("Generating training data from one self-play game...")
    states, policy_targets, value = self_play_game(net, num_simulations=50)
    
    print(f"\nGame finished. Winner value: {value}")
    print(f"Generated {len(states)} training examples.")

    # Now, you would typically collect data from many games and then train the network.
    # For demonstration, we'll just show the shapes of the generated data.
    
    # 1. Convert states to a tensor
    state_tensors = np.array([Neuralnet._state_to_tens(s).numpy() for s in states])
    print(f"States tensor shape: {state_tensors.shape}")

    # 2. Convert policy targets to a tensor
    policy_targets_tensor = np.array(policy_targets, dtype=np.float32)
    print(f"Policy targets tensor shape: {policy_targets_tensor.shape}")

    # 3. Create value targets tensor
    value_targets_tensor = np.full(len(states), value, dtype=np.float32)
    print(f"Value targets tensor shape: {value_targets_tensor.shape}")

    # The next step would be to pass these tensors to a `net.train()` method.
    print("\nNext step: Implement and call neuralnet.train() with this data.")
