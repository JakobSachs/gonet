from typing import List, Optional, Set, Tuple

import numpy as np


class Board:
    EMPTY = 0
    BLACK = 1
    WHITE = 2
    SIZE = 9

    def __init__(self) -> None:
        self.board: np.ndarray = np.full((self.SIZE, self.SIZE), self.EMPTY, dtype=int)

    def display(self) -> None:
        symbols = {self.EMPTY: ".", self.BLACK: "X", self.WHITE: "O"}
        print("  " + " ".join(str(i) for i in range(self.SIZE)))
        for y in range(self.SIZE):
            row = [symbols[self.board[y, x]] for x in range(self.SIZE)]
            print(f"{y} " + " ".join(row))

    def is_on_board(self, x: int, y: int) -> bool:
        return 0 <= x < self.SIZE and 0 <= y < self.SIZE

    def _has_liberty(
        self,
        x: int,
        y: int,
        board: np.ndarray,
        player: int,
        visited: Optional[Set[Tuple[int, int]]] = None,
    ) -> bool:
        if visited is None:
            visited = set()
        if (x, y) in visited:
            return False
        visited.add((x, y))
        for nx, ny in self._neighbors(x, y):
            if self.is_on_board(nx, ny):
                if board[ny, nx] == self.EMPTY:
                    return True
                if board[ny, nx] == player and self._has_liberty(
                    nx, ny, board, player, visited
                ):
                    return True
        return False

    def _neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        return [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

    def _get_group(
        self,
        x: int,
        y: int,
        player: int,
        visited: Optional[Set[Tuple[int, int]]] = None,
    ) -> List[Tuple[int, int]]:
        # find connected group
        if visited is None:
            visited = set()
        group: List[Tuple[int, int]] = []
        stack: List[Tuple[int, int]] = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            group.append((cx, cy))
            for nx, ny in self._neighbors(cx, cy):
                if (
                    self.is_on_board(nx, ny)
                    and self.board[ny, nx] == player
                    and (nx, ny) not in visited
                ):
                    stack.append((nx, ny))
        return group

    def _explore_region(self, x: int, y: int, visited: Set[Tuple[int, int]]):
        """Find all empty points in a region and the colors bordering it."""
        region = []
        border_colors = set()
        stack = [(x, y)]
        region_visited = set()

        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in region_visited:
                continue

            region_visited.add((cx, cy))
            region.append((cx, cy))

            for nx, ny in self._neighbors(cx, cy):
                if not self.is_on_board(nx, ny):
                    border_colors.add("EDGE")  # Mark that the region touches the edge
                    continue

                neighbor_val = self.board[ny, nx]
                if neighbor_val == self.EMPTY:
                    if (nx, ny) not in region_visited:
                        stack.append((nx, ny))
                else:  # It's a player stone
                    border_colors.add(neighbor_val)

        visited.update(region_visited)
        return region, border_colors

    def _get_group_with_liberty_sim(
        self,
        x: int,
        y: int,
        player: int,
        board: np.ndarray,
        visited: Set[Tuple[int, int]],
    ) -> Tuple[List[Tuple[int, int]], bool]:
        group = []
        has_liberty = False
        stack = [(x, y)]
        group_visited = set()

        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in group_visited:
                continue

            group_visited.add((cx, cy))
            visited.add((cx, cy))
            group.append((cx, cy))

            for nx, ny in self._neighbors(cx, cy):
                if not self.is_on_board(nx, ny):
                    continue

                neighbor_stone = board[ny, nx]
                if neighbor_stone == self.EMPTY:
                    has_liberty = True
                elif neighbor_stone == player and (nx, ny) not in group_visited:
                    stack.append((nx, ny))
        return group, has_liberty


def print_board(board: "Board") -> None:
    """Nicely print the board state to the terminal using unicode symbols."""
    symbols = {
        board.EMPTY: ".",
        board.WHITE: "\u25cf",  # ●
        board.BLACK: "\u25cb",  # ○
    }
    # The unicode characters for stones are often wider than a single character.
    # We add an extra space after each symbol to ensure alignment.
    print("   " + "  ".join(map(str, range(board.SIZE))))
    for y in range(board.SIZE):
        row = "  ".join(symbols[board.board[y, x]] for x in range(board.SIZE))
        print(f"{y:2d} {row}")
