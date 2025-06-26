import random

from game import Game


def main():
    game = Game()
    human_player = game.board.BLACK  # Human is always Black, goes first
    cpu_player = game.board.WHITE

    while not game.game_over:
        game.display_board()
        if game.current_player == human_player:
            print(f"Your turn (Black).")
            try:
                move = input("Enter your move as 'x y', 'pass', or 'quit': ").lower()
                if move == "quit":
                    break
                if move == "pass":
                    game.pass_turn()
                    continue
                x_str, y_str = move.split()
                x, y = int(x_str), int(y_str)
                if not game.place_stone(x, y):
                    print("Invalid move, try again.")
            except ValueError:
                print("Invalid input. Please enter 'x y', 'pass', or 'quit'.")
        else:
            # CPU's turn (random move)
            legal_moves = game.legal_moves()
            if legal_moves:
                move = random.choice(legal_moves)
                print(f"CPU (White) plays: {move[0]} {move[1]}")
                game.place_stone(move[0], move[1])
            else:
                print("CPU (White) passes.")
                game.pass_turn()


if __name__ == "__main__":
    main()
