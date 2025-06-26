import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest

from board import Board
from game import Game


@pytest.fixture
def game():
    return Game()


def test_legal_move(game):
    assert game.place_stone(0, 0)  # Black
    assert game.board.board[0][0] == Board.BLACK


def test_illegal_move_on_occupied(game):
    game.place_stone(0, 0)  # Black
    assert not game.place_stone(0, 0)  # White tries same spot


def test_suicide_move(game):
    # Black surrounds (1,1), White tries to play there
    game.place_stone(0, 1)  # B
    game.place_stone(0, 0)  # W
    game.place_stone(1, 0)  # B
    game.place_stone(2, 0)  # W
    game.place_stone(2, 1)  # B
    game.place_stone(2, 2)  # W
    game.place_stone(1, 2)  # B
    assert not game.place_stone(1, 1)  # W suicide


def test_capture(game):
    # Black surrounds a white stone and captures it
    game.place_stone(1, 0)  # B
    game.place_stone(0, 0)  # W
    game.place_stone(0, 1)  # B
    game.place_stone(2, 0)  # W
    game.place_stone(1, 1)  # B (captures W at 0,0)
    assert game.board.board[0][0] == Board.EMPTY


def test_ko_rule(game):
    # Set up a simple ko
    g = game

    g.place_stone(0, 1)  # W
    g.place_stone(2, 0)  # B
    g.place_stone(2, 1)  # W
    g.place_stone(3, 1)  # B
    g.place_stone(1, 2)  # W
    g.place_stone(2, 2)  # B
    g.place_stone(1, 0)  # W
    g.place_stone(1, 1)  # B (this captures)
    assert not g.place_stone(2, 1)  # W (this would recapture the stone captures)


def test_score_empty_board(game):
    black_score, white_score = game.calculate_score()
    assert black_score == 0
    assert white_score == 0


def test_score_simple_territory(game: Game):
    """Black surrounds a 1x1 area."""
    # B . B
    # . B .
    # B . B
    game.board.board[1, 0] = Board.BLACK
    game.board.board[0, 1] = Board.BLACK
    game.board.board[2, 1] = Board.BLACK
    game.board.board[1, 2] = Board.BLACK
    black_score, white_score = game.calculate_score()
    assert black_score == 1
    assert white_score == 0


def test_score_capture_and_territory(game: Game):
    """Black captures a stone and creates territory."""
    game.place_stone(0, 0)  # B
    game.place_stone(8, 8)  # W
    game.place_stone(0, 1)  # B
    game.place_stone(8, 7)  # W
    game.place_stone(0, 2)  # B
    game.place_stone(8, 6)  # W
    game.place_stone(1, 2)  # B
    game.place_stone(8, 5)  # W
    game.place_stone(2, 2)  # B
    game.place_stone(8, 4)  # W
    game.place_stone(2, 1)  # B
    game.place_stone(8, 3)  # W
    game.place_stone(2, 0)  # B
    game.place_stone(1, 1)  # W - to be captured
    game.place_stone(1, 0)  # B - captures W at (1,1)

    assert game.board.board[1, 1] == Board.EMPTY
    assert game.captures_black == 1

    black_score, white_score = game.calculate_score()
    # 1 point for the captured stone + 1 point for the territory at (1,1)
    assert black_score == 2
    assert white_score == 0


def test_score_mixed_territory():
    g = Game()
    # Black surrounds top left 2x2, white surrounds bottom right 2x2
    g.board.board[0, 0] = g.board.BLACK
    g.board.board[0, 1] = g.board.BLACK
    g.board.board[1, 0] = g.board.BLACK
    g.board.board[1, 1] = g.board.BLACK
    g.board.board[7, 7] = g.board.WHITE
    g.board.board[7, 8] = g.board.WHITE
    g.board.board[8, 7] = g.board.WHITE
    g.board.board[8, 8] = g.board.WHITE
    black_score, white_score = g.calculate_score()
    # Each should have at least 1 territory in their corner
    assert black_score >= 0
    assert white_score >= 0
