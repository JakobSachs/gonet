from typing import List, Optional, Set, Tuple

import numpy as np

from board import Board, print_board


class Game:
    def __init__(self) -> None:
        self.board = Board()
        self.current_player: int = self.board.BLACK
        self.previous_board: Optional[np.ndarray] = None  # For ko rule
        self.captures_black: int = 0
        self.captures_white: int = 0
        self.game_over = False
        self.passes = 0

    def place_stone(self, x: int, y: int) -> bool:
        if self.game_over:
            print("Game is over.")
            return False

        if not self.is_legal_move(x, y, self.current_player):
            print("ILLEGAL MOVE", x, y)
            return False

        self.previous_board = self.board.board.copy()
        self.board.board[y, x] = self.current_player
        captured_stones = self._remove_captured_stones(
            self.get_opponent(self.current_player)
        )

        if self.current_player == self.board.BLACK:
            self.captures_black += captured_stones
        else:
            self.captures_white += captured_stones

        self.current_player = self.get_opponent(self.current_player)
        self.passes = 0
        return True

    def is_legal_move(self, x: int, y: int, player: int) -> bool:
        if not self.board.is_on_board(x, y):
            return False
        if self.board.board[y, x] != self.board.EMPTY:
            return False

        test_board_state = self.board.board.copy()
        test_board_state[y, x] = player
        opponent = self.get_opponent(player)

        # Simulate captures
        captured_in_sim = self._get_captured_stones_on_sim_board(
            opponent, test_board_state
        )
        for tx, ty in captured_in_sim:
            test_board_state[ty, tx] = self.board.EMPTY

        # Ko rule
        if self.previous_board is not None and np.array_equal(
            test_board_state, self.previous_board
        ):
            print("that would be KO")
            return False

        # Check for liberties (suicide rule)
        _, has_liberty = self.board._get_group_with_liberty_sim(
            x, y, player, test_board_state, set()
        )
        if not has_liberty:
            # If no liberties, check if it's a capture move that saves it from suicide
            if not captured_in_sim:
                return False

        return True

    def _get_captured_stones_on_sim_board(
        self, player: int, sim_board: np.ndarray
    ) -> List[Tuple[int, int]]:
        to_remove = []
        visited = set()
        for ty in range(self.board.SIZE):
            for tx in range(self.board.SIZE):
                if sim_board[ty, tx] == player and (tx, ty) not in visited:
                    group, has_liberty = self.board._get_group_with_liberty_sim(
                        tx, ty, player, sim_board, visited
                    )
                    if not has_liberty:
                        to_remove.extend(group)
        return to_remove

    def _remove_captured_stones(self, player: int) -> int:
        to_remove = []
        visited = set()
        for y in range(self.board.SIZE):
            for x in range(self.board.SIZE):
                if self.board.board[y, x] == player and (x, y) not in visited:
                    if not self.board._has_liberty(
                        x, y, self.board.board, player, set()
                    ):
                        group = self.board._get_group(x, y, player, visited)
                        to_remove.extend(group)

        for x, y in to_remove:
            self.board.board[y, x] = self.board.EMPTY
        return len(to_remove)

    def pass_turn(self) -> None:
        if self.game_over:
            print("Game is over.")
            return
        self.previous_board = self.board.board.copy()
        self.current_player = self.get_opponent(self.current_player)
        self.passes += 1
        print(f"Player {self.get_opponent(self.current_player)} passes.")
        if self.passes >= 2:
            self.end_game()

    def end_game(self) -> None:
        self.game_over = True
        black_score, white_score = self.calculate_score()
        print("Game Over")
        print(f"Black Score: {black_score}")
        print(f"White Score: {white_score}")
        if black_score > white_score:
            print("Black Wins!")
        elif white_score > black_score:
            print("White Wins!")
        else:
            print("It's a Tie!")

    def calculate_score(self) -> Tuple[int, int]:
        """Calculate Japanese/Korean (territory) score for Black and White."""
        visited: Set[Tuple[int, int]] = set()
        territory_black = 0
        territory_white = 0

        for y in range(self.board.SIZE):
            for x in range(self.board.SIZE):
                if self.board.board[y, x] == self.board.EMPTY and (x, y) not in visited:
                    region, border_colors = self.board._explore_region(x, y, visited)
                    if "EDGE" in border_colors:
                        continue  # Not a valid territory if it touches the edge

                    if len(border_colors) == 1:
                        color = border_colors.pop()
                        if color == self.board.BLACK:
                            territory_black += len(region)
                        else:  # color == self.board.WHITE
                            territory_white += len(region)

        score_black = territory_black + self.captures_black
        score_white = territory_white + self.captures_white
        return score_black, score_white

    def get_opponent(self, player: int) -> int:
        return self.board.BLACK if player == self.board.WHITE else self.board.WHITE

    def legal_moves(self) -> List[Tuple[int, int]]:
        moves: List[Tuple[int, int]] = []
        for y in range(self.board.SIZE):
            for x in range(self.board.SIZE):
                if self.is_legal_move(x, y, self.current_player):
                    moves.append((x, y))
        return moves

    def display_board(self) -> None:
        print_board(self.board)
