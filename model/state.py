import enum
from dataclasses import dataclass

import numpy as np


class Stone(enum.Enum):
    WHITE = 1
    BLACK = 2
    EMPTY = 0


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
    def empty() -> 'Board':
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
    def empty() -> 'Gamestate':
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

        liberties: set[tuple[int, int]] = set()
        stack = [(x, y)]
        visited: set[tuple[int, int]] = set()

        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < 9 and 0 <= ny < 9:
                    neighbor_stone = board[nx, ny]
                    if neighbor_stone == stone_color:
                        if (nx, ny) not in visited:
                            stack.append((nx, ny))
                    elif neighbor_stone == Stone.EMPTY.value:
                        liberties.add((nx, ny))
        return visited, liberties

    def do_move(self, move: tuple[int, int]) -> 'Gamestate':
        if move == (-1, -1):
            return Gamestate(
                previous=self.board,
                board=self.board,
                blacks_points=self.blacks_points,
                blacks_turn=not self.blacks_turn,
                last_move_was_pass=True,
                game_over=self.last_move_was_pass,
            )
        x, y = move
        if self.board[x, y] != Stone.EMPTY.value:
            raise ValueError("Position is not empty")

        new_board = self.board.copy()
        stone_color = Stone.BLACK.value if self.blacks_turn else Stone.WHITE.value
        new_board[x, y] = stone_color

        opponent_color = Stone.WHITE.value if self.blacks_turn else Stone.BLACK.value
        captured_stones = 0

        def neighbors(x, y):
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 9 and 0 <= ny < 9:
                    yield nx, ny

        for nx, ny in neighbors(x, y):
            if new_board[nx, ny] == opponent_color:
                group, liberties = self._get_group(new_board, nx, ny)
                if not liberties:
                    captured_stones += len(group)
                    for gx, gy in group:
                        new_board[gx, gy] = Stone.EMPTY.value

        # Check for suicide
        my_group, my_liberties = self._get_group(new_board, x, y)
        if not my_liberties and captured_stones == 0:
            raise ValueError("Illegal suicide move")

        # Check for Ko
        if np.array_equal(new_board, self.previous):
            raise ValueError("Ko rule violation")

        new_blacks_points = self.blacks_points
        if self.blacks_turn:
            new_blacks_points += captured_stones
        else:
            new_blacks_points -= captured_stones

        return Gamestate(
            previous=self.board,
            board=new_board,
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
                    try:
                        self.do_move((x, y))
                        moves.append((x, y))
                    except ValueError:
                        # Illegal move (suicide or Ko)
                        pass
        return moves
