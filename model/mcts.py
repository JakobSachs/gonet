import math, copy
import numpy as np

class MCTSNode:
    __slots__ = ("prior","Q","N","children")
    def __init__(self, prior):
        self.prior = prior
        self.Q = 0.0
        self.N = 0
        self.children = {}  # move -> MCTSNode

class MCTS:
    def __init__(self, model, c_puct=1.0, sims=100):
        self.model = model.eval()
        self.c_puct = c_puct
        self.sims = sims

    def run(self, root_state):
        root = MCTSNode(prior=1.0)
        for _ in range(self.sims):
            self._simulate(root, root_state.clone())
        # return visit‐counts π
        pi = np.zeros(root_state.num_moves())
        for mv, node in root.children.items():
            pi[mv] = node.N
        return pi / pi.sum()

    def _simulate(self, node, state):
        if state.is_terminal():
            z = state.reward()  # +1/-1
        elif not node.children:
            # expand
            policy = state.legal_moves_uniform()  # uniform prior
            for mv in policy:
                node.children[mv] = MCTSNode(prior=1.0/len(policy))
            # leaf eval
            with torch.no_grad():
                x = state.to_tensor().unsqueeze(0)
                z = float(self.model(x).item())
        else:
            # select
            best_mv, best_child = max(
                node.children.items(),
                key=lambda kv: (
                    kv[1].Q/kv[1].N if kv[1].N>0 else 0
                    + self.c_puct*kv[1].prior *
                      math.sqrt(sum(c.N for c in node.children.values())+1)/(1+kv[1].N)
                )
            )
            state.play(best_mv)
            z = self._simulate(best_child, state)
        # backprop
        node.N += 1
        node.Q += (z - node.Q) / node.N
        return -z
