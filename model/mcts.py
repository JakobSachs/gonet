from state import Gamestate, Stone, Board
from neuralnet import Valuenet

from dataclasses import dataclass, field
from typing import Self, Any
import math
import random


import os
from tqdm import tqdm
import numpy as np


DEBUG = os.environ.get("DEBUG") == "1"


@dataclass
class MCTSNode:
    state: Gamestate
    parent: Self | None
    move: Any | None  # the move that led here (None for root)
    visits: int = 0
    value: float = 0.0  # sum of scores from rollouts
    children: list[Self] = field(default_factory=list)
    depth: int = field(init=False)  # depth of this node in the search tree
    untried_moves: list = field(init=False)

    def __post_init__(self):
        self.untried_moves = self.state.get_moves()
        # Calculate depth based on parent
        if self.parent is None:
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1

    def uct_score(self, parent_visits: int) -> float:
        if self.visits == 0:
            return math.inf  # Ensure unvisited nodes are selected first
        # UCT formula
        exploitation = self.value / self.visits
        exploration = math.sqrt(2) * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration


def mcts_neural(
    value_net: Valuenet,
    root: MCTSNode,
    iterations: int = 100,
) -> MCTSNode:

    # each iteration checks a possible node in the tree
    for _ in range(iterations):
        # traverse tree till we find a note to expand, ordering based on UCT-score
        node = root
        while (
            not node.state.game_over and len(node.untried_moves) == 0 and node.children
        ):
            # choose child with highest UCT
            node = max(node.children, key=lambda n: n.uct_score(node.visits))

        # expand chosen note by randomly choosing a child
        if not node.state.game_over and node.untried_moves:
            m = node.untried_moves.pop(random.randrange(len(node.untried_moves)))
            next_state = node.state.do_move(m)
            child = MCTSNode(state=next_state, parent=node, move=m)
            node.children.append(child)
            node = child

        # ROLLOUT/Q-value guessing 
        pred_value = value_net.predict(node.state)
        canonical_reward = pred_value + node.state.blacks_points
        if not node.state.blacks_turn:
            canonical_reward = -canonical_reward

        # propagate resulting value up the search-tree
        while node is not None:
            node.visits += 1
            if node.state.blacks_turn:
                node.value += canonical_reward
            else:
                node.value -= canonical_reward
            node = node.parent

    children = root.children[:]
    children.sort(key=lambda n: n.visits, reverse=True)

    # select best move based on visits (opposed to value) to make sure we dont accidentally choose an under-explored
    # branch with seemingly good value
    best_child = max(root.children, key=lambda n: n.visits)
    return best_child



def play_game(
    value_net: Valuenet, iterations_per_move: int = 100
) -> list[tuple[Gamestate, int]]:
    """
    Plays a full game of self-play using MCTS, recording game states.

    Returns:
        A list of tuples, where each tuple contains a game state and the final
        outcome of the game (1 for black win, -1 for white win, 0 for draw).
        This data is crucial for training the neural network.
    """
    game_states = []
    state = Gamestate.empty()
    root = MCTSNode(state=state, parent=None, move=None)

    while not root.state.game_over:
        game_states.append(root.state)
        best_child = mcts_neural(value_net, root, iterations=iterations_per_move)
        best_child.parent = None  # The new state becomes the root for the next turn
        root = best_child

    # Determine the game outcome
    final_score = root.state.score()
    if final_score > 0:
        outcome = 1  # Black wins
    elif final_score < 0:
        outcome = -1  # White wins
    else:
        outcome = 0  # Draw

    # Grade each state with the final outcome
    training_data = [(s, outcome) for s in game_states]
    return training_data


if __name__ == "__main__":
    net = Valuenet()
    # selfplay
    for loopi in range(100):
        print("play-train loop: ", loopi)
        num_games_to_play = 30
        all_training_samples = []
        print(
            f"Starting self-play to generate training data for {num_games_to_play} games..."
        )
        for i in tqdm(range(num_games_to_play)):
            # In self-play, we can use fewer MCTS iterations to generate games faster.
            training_samples = play_game(net, iterations_per_move=150)
            all_training_samples.extend(training_samples)

        print(f"\nGenerated {len(all_training_samples)} training samples.")

        states, outcomes = zip(*all_training_samples)
        net.train(states, outcomes)
        net.save("trained_model.pth")
