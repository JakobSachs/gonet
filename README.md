## Model

- **PolicyNet**: Neural network estimating move probabilities (policy) for a given board state.
- **ValueNet**: Neural network estimating the value (expected outcome) of a board state.
- **DualNetGoBot**: Combines PolicyNet and ValueNet for reinforcement learning; selects moves using the policy net and evaluates positions with the value net. 