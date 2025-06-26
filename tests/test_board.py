import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from board import Board


@pytest.fixture
def board():
    """Returns a Board instance."""
    return Board()


def test_board_initialization(board: Board):
    """Test that the board is initialized correctly."""
    assert board.board.shape == (Board.SIZE, Board.SIZE)
    assert np.all(board.board == Board.EMPTY)
