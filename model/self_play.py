import torch
import numpy as np

def self_play_worker(
    wid: int,
    model,
    train_queue,
    event_queue,
    cfg
):
    torch.manual_seed(cfg.seed + wid)
    while True:
        game = GoGame(cfg.board_size)
        mcts = MCTS(model, cfg.c_puct, cfg.sims)
        trajectory = []
        # play full game
        while not game.is_over():
            pi = mcts.run(game)
            move = np.random.choice(len(pi), p=pi)
            trajectory.append((game.to_tensor(), pi, game.current_player))
            game.play(move)
            # emit move for viewer
            event_queue.put({
                "type": "move",
                "game_id": f"{wid}-{game.move_count}",
                "move": move,
                "board": game.serialize_board()
            })
        # game finished
        winner = game.winner()  # +1 or -1
        # send training samples
        for state, pi, player in trajectory:
            z = 1 if player == winner else -1
            train_queue.put((state, pi, z))
        event_queue.put({
            "type": "game_end",
            "game_id": f"{wid}-{game.move_count}",
            "winner": winner
        })
