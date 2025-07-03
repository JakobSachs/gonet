import sys
import os
from state import Gamestate, Stone
from neuralnet import Valuenet
from tinygrad.nn.state import safe_load

# Unicode characters for Go board rendering
STONE_BLACK = "●"
STONE_WHITE = "○"
STONE_EMPTY = "·"


def render_board(state: Gamestate):
    board = state.board
    size = board.shape[0]
    print("   " + " ".join(chr(ord('A') + i) for i in range(size)))
    for i in range(size):
        row = []
        for j in range(size):
            if board[i, j] == Stone.BLACK.value:
                row.append(STONE_BLACK)
            elif board[i, j] == Stone.WHITE.value:
                row.append(STONE_WHITE)
            else:
                row.append(STONE_EMPTY)
        print(f"{i+1:2d} " + " ".join(row))
    print()


def parse_move(move_str, size):
    move_str = move_str.strip().upper()
    if move_str == "QUIT":
        return "QUIT"
    if move_str == "PASS":
        return (-1, -1)
    if len(move_str) < 2:
        return None
    col = ord(move_str[0]) - ord('A')
    try:
        row = int(move_str[1:]) - 1
    except ValueError:
        return None
    if 0 <= row < size and 0 <= col < size:
        return (row, col)
    return None


def main():
    model_path = os.path.join(os.path.dirname(__file__), "trained_model.pth")
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)
    net = Valuenet()
    # Load model weights if available
    if os.path.exists(model_path):
        state_dict = safe_load(model_path)
        net.__dict__.update(state_dict)

    state = Gamestate.empty()
    print("Welcome to Go! You are playing against the AI.")
    print("Enter moves in the format 'A1', 'D4', etc. Type 'pass' to pass.")
    print()
    human_is_black = input("Do you want to play as black? (y/n): ").strip().lower() == 'y'

    while not state.game_over:
        render_board(state)
        if (state.blacks_turn and human_is_black) or (not state.blacks_turn and not human_is_black):
            # Human move
            moves = state.get_moves()
            while True:
                move_str = input(f"Your move ({'Black' if state.blacks_turn else 'White'}): ")
                move = parse_move(move_str, state.board.shape[0])
                if move == "QUIT":
                    print("Quitting the game. Goodbye!")
                    return
                if move == (-1, -1) and (-1, -1) in moves:
                    state = state.do_move((-1, -1))
                    break
                if move is not None and move in moves:
                    state = state.do_move(move)
                    break
                print("Invalid move. Try again.")
        else:
            # AI move
            from mcts import MCTSNode, mcts_neural
            root = MCTSNode(state=state, parent=None, move=None)
            print("AI is thinking...")
            best_child = mcts_neural(net, root, iterations=200)
            move = best_child.move
            if move == (-1, -1):
                move_str = "PASS"
            elif move is not None:
                move_str = chr(ord('A')+move[1]) + str(move[0]+1)
            else:
                move_str = "?"
            print(f"AI plays: {move_str}")
            if move is not None:
                state = state.do_move(move)
            else:
                print("AI did not return a valid move. Exiting.")
                break

    render_board(state)
    final_score = state.score()
    print(f"Game over! Final score: {final_score}")
    if final_score > 0:
        print("Black wins!")
    elif final_score < 0:
        print("White wins!")
    else:
        print("It's a draw!")

if __name__ == "__main__":
    main() 