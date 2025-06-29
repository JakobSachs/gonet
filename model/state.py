import numpy as np
from typing import Self
import enum
from dataclasses import dataclass


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
        # map 0→· (empty) , 1→● (black) , 2→○ (white)
        symbols = {0: "·", 1: "●", 2: "○"}
        lines = []
        for row in self.tolist():
            # join with spaces so stones line up
            lines.append(" ".join(symbols[val] for val in row))
        return "\n".join(lines)

    @staticmethod
    def empty() -> Self:
        return Board(np.zeros((9, 9), dtype=int))


@dataclass
class Gamestate:
    previous: Board
    board: Board
    points: tuple[float, float]  # (black, white)
    blacks_turn: bool  # whether it's black-turn

    @staticmethod
    def empty() -> Self:
        return Gamestate(
            previous=Board.empty(), board=Board.empty(), points=(0, 6.5), blacks_turn=True
        )

    

print(Gamestate.empty())
