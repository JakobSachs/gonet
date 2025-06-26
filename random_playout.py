import random

from game import Game

if __name__ == "__main__":
    game = Game()
    move_num = 1
    while not game.game_over:
        legal_moves = game.legal_moves()
        # Randomly decide to pass (10% chance) or if no legal moves, must pass
        should_pass = (random.random() < 0.1) or not legal_moves
        if should_pass:
            print(f"Move {move_num}: Player {game.current_player} passes.")
            game.pass_turn()
        else:
            move = random.choice(legal_moves)
            print(f"Move {move_num}: Player {game.current_player} plays {move}")
            game.place_stone(*move)
        move_num += 1

    print("Game over.")
    game.display_board()
    # Print final score
    black_score, white_score = game.calculate_score()
    print("Final Score:")
    print(f"Black: {black_score}")
    print(f"White: {white_score}")
