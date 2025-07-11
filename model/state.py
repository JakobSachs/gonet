import enum
from dataclasses import dataclass

import numpy as np
from numba import jit
from numba.core import types
from numba.typed import List

# Numba-friendly constants and types
int_tuple = types.UniTuple(types.int64, 2)
BLACK_STONE = 2
WHITE_STONE = 1
EMPTY_STONE = 0


@jit(nopython=True)
def _get_group_jit(
    board: np.ndarray, x: int, y: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Finds the group of connected stones of the same color and their liberties.
    Numba-jitted version.
    """
    stone_color = board[x, y]
    if stone_color == EMPTY_STONE:
        return List.empty_list(int_tuple), List.empty_list(int_tuple)

    group = List.empty_list(int_tuple)
    liberties = List.empty_list(int_tuple)
    stack = List([(x, y)])

    visited_group = np.zeros_like(board, dtype=np.bool_)
    visited_group[x, y] = True

    visited_liberties = np.zeros_like(board, dtype=np.bool_)

    while len(stack) > 0:
        cx, cy = stack.pop()
        group.append((cx, cy))

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < 9 and 0 <= ny < 9:
                neighbor_stone = board[nx, ny]
                if neighbor_stone == stone_color:
                    if not visited_group[nx, ny]:
                        visited_group[nx, ny] = True
                        stack.append((nx, ny))
                elif neighbor_stone == EMPTY_STONE:
                    if not visited_liberties[nx, ny]:
                        visited_liberties[nx, ny] = True
                        liberties.append((nx, ny))
    return group, liberties


@jit(nopython=True)
def _apply_move_jit(
    board: np.ndarray, move: tuple[int, int], blacks_turn: bool
) -> tuple[np.ndarray, int]:
    """
    Applies a move to the board and returns the new board and captured stones.
    Does not check for legality.
    """
    x, y = move
    new_board = board.copy()
    stone_color = BLACK_STONE if blacks_turn else WHITE_STONE
    new_board[x, y] = stone_color

    opponent_color = WHITE_STONE if blacks_turn else BLACK_STONE
    captured_stones = 0

    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 9 and 0 <= ny < 9:
            if new_board[nx, ny] == opponent_color:
                group, liberties = _get_group_jit(new_board, nx, ny)
                if len(liberties) == 0 and len(group) > 0:
                    captured_stones += len(group)
                    for gx, gy in group:
                        new_board[gx, gy] = EMPTY_STONE

    return new_board, captured_stones


@jit(nopython=True)
def _is_move_legal_jit(
    board: np.ndarray,
    previous_board: np.ndarray,
    move: tuple[int, int],
    blacks_turn: bool,
) -> bool:
    """
    Checks if a move is legal (not suicide, not Ko).
    Numba-jitted version.
    """
    x, y = move
    if board[x, y] != EMPTY_STONE:
        return False

    new_board, captured_stones = _apply_move_jit(board, move, blacks_turn)

    # Check for suicide
    my_group, my_liberties = _get_group_jit(new_board, x, y)
    if len(my_liberties) == 0 and captured_stones == 0:
        return False  # Illegal suicide move

    # Check for Ko
    if np.array_equal(new_board, previous_board):
        return False  # Ko rule violation

    return True


class Stone(enum.Enum):
    WHITE = WHITE_STONE
    BLACK = BLACK_STONE
    EMPTY = EMPTY_STONE


class Board(np.ndarray):
    def __new__(cls, input_array):
        arr = np.asarray(input_array, dtype=int).view(cls)
        if arr.shape != (9, 9):
            raise ValueError("Board must be 9×9")
        return arr

    def __str__(self):
        symbols = {0: "·", 1: "●", 2: "○"}
        lines = []
        for row in self.tolist():
            # join with spaces so stones line up
            lines.append(" ".join(symbols[val] for val in row))
        return "\n".join(lines)

    @staticmethod
    def empty() -> "Board":
        return Board(np.zeros((9, 9), dtype=int))


@dataclass
class Gamestate:
    previous: Board
    board: Board
    blacks_points: float  # the extra points gained/lost by black through capture/losses
    blacks_turn: bool  # whether it's black-turn
    last_move_was_pass: bool
    game_over: bool

    @staticmethod
    def empty() -> "Gamestate":
        return Gamestate(
            previous=Board.empty(),
            board=Board.empty(),
            blacks_points=0,  # Komi is handled by the score function
            blacks_turn=True,
            last_move_was_pass=False,
            game_over=False,
        )

    def score(self) -> float:
        visited = np.zeros((9, 9), dtype=bool)
        black_territory, white_territory = 0, 0

        def neighbors(x, y):
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 9 and 0 <= ny < 9:
                    yield nx, ny

        def flood_fill(x, y):
            stack = [(x, y)]
            region = set()
            neighbor_colors = set()

            while stack:
                cx, cy = stack.pop()
                if visited[cx, cy]:
                    continue
                visited[cx, cy] = True
                region.add((cx, cy))

                for nx, ny in neighbors(cx, cy):
                    val = self.board[nx, ny]
                    if val == Stone.EMPTY.value and not visited[nx, ny]:
                        stack.append((nx, ny))
                    elif val in (Stone.BLACK.value, Stone.WHITE.value):
                        neighbor_colors.add(val)

            return region, neighbor_colors

        for x in range(9):
            for y in range(9):
                if self.board[x, y] == Stone.EMPTY.value and not visited[x, y]:
                    region, neighbor_colors = flood_fill(x, y)
                    if neighbor_colors == {Stone.BLACK.value}:
                        black_territory += len(region)
                    elif neighbor_colors == {Stone.WHITE.value}:
                        white_territory += len(region)
                    # else: neutral, ignore

        black_stones = np.count_nonzero(self.board == Stone.BLACK.value)
        white_stones = np.count_nonzero(self.board == Stone.WHITE.value)

        black_score = black_territory + black_stones + self.blacks_points
        white_score = white_territory + white_stones + 6.5  # Komi

        return black_score - white_score

    def _get_group(
        self, board: Board, x: int, y: int
    ) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        """
        Finds the group of connected stones of the same color and their liberties.
        """
        stone_color = board[x, y]
        if stone_color == Stone.EMPTY.value:
            return set(), set()

        group_list, liberties_list = _get_group_jit(board, x, y)
        return set(group_list), set(liberties_list)

    def do_move(self, move: tuple[int, int]) -> "Gamestate":
        if move == (-1, -1):
            return Gamestate(
                previous=self.board,
                board=self.board,
                blacks_points=self.blacks_points,
                blacks_turn=not self.blacks_turn,
                last_move_was_pass=True,
                game_over=self.last_move_was_pass,
            )

        if not _is_move_legal_jit(self.board, self.previous, move, self.blacks_turn):
            raise ValueError("Illegal move")

        new_board_arr, captured_stones = _apply_move_jit(
            self.board, move, self.blacks_turn
        )

        new_blacks_points = self.blacks_points
        if self.blacks_turn:
            new_blacks_points += captured_stones
        else:
            new_blacks_points -= captured_stones

        return Gamestate(
            previous=self.board,
            board=Board(new_board_arr),
            blacks_points=new_blacks_points,
            blacks_turn=not self.blacks_turn,
            last_move_was_pass=False,
            game_over=False,
        )

    def get_moves(self) -> list[tuple[int, int]]:
        moves = [(-1, -1)]
        for x in range(9):
            for y in range(9):
                if self.board[x, y] == Stone.EMPTY.value:
                    if _is_move_legal_jit(
                        self.board, self.previous, (x, y), self.blacks_turn
                    ):
                        moves.append((x, y))
        return moves
