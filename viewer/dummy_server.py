
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "websockets",
# ]
# ///

import asyncio
import json
import random
import websockets

# --- Configuration ---
BOARD_COUNT = 16
GRID_SIZE = 9
COLORS = [1, 2]  # 1 for black, 2 for white
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8765
MOVE_LIMIT = 100  # Max moves per board before reset

# --- Server State ---
GAME_STATES = [[0] * (GRID_SIZE * GRID_SIZE) for _ in range(BOARD_COUNT)]
MOVE_COUNTS = [0] * BOARD_COUNT
SUBSCRIBED_CLIENTS = set()

def get_initial_state_message():
    """Creates a message containing the full state of all games."""
    return json.dumps({"type": "initial_state", "payload": GAME_STATES})

def get_move_update_message(board_idx, x, y, color):
    """Creates a message for a single move update."""
    return json.dumps({
        "type": "move_update",
        "payload": {"board_idx": board_idx, "x": x, "y": y, "color": color},
    })

def get_board_reset_message(board_idx):
    """Creates a message to signal that a board has been reset."""
    return json.dumps({"type": "board_reset", "payload": {"board_idx": board_idx}})

async def move_producer():
    """Periodically generates a new move and broadcasts it to all clients."""
    while True:
        await asyncio.sleep(0.01)
        if not SUBSCRIBED_CLIENTS:
            continue

        board_idx = random.randint(0, BOARD_COUNT - 1)

        # Check if the board has reached its move limit
        if MOVE_COUNTS[board_idx] >= MOVE_LIMIT:
            print(f"Board {board_idx} reached {MOVE_LIMIT} moves. Resetting.")
            # Reset the board state and move count
            GAME_STATES[board_idx] = [0] * (GRID_SIZE * GRID_SIZE)
            MOVE_COUNTS[board_idx] = 0
            # Notify clients of the reset
            reset_message = get_board_reset_message(board_idx)
            websockets.broadcast(SUBSCRIBED_CLIENTS, reset_message)
            continue

        # 1. Generate a random move
        x = random.randint(0, GRID_SIZE - 1)
        y = random.randint(0, GRID_SIZE - 1)
        color = random.choice(COLORS)

        # 2. Update server state
        GAME_STATES[board_idx][y * GRID_SIZE + x] = color
        MOVE_COUNTS[board_idx] += 1

        # 3. Create and broadcast the update message
        message = get_move_update_message(board_idx, x, y, color)
        websockets.broadcast(SUBSCRIBED_CLIENTS, message)
        # print(f"Broadcasted move to {len(SUBSCRIBED_CLIENTS)} clients.")


async def connection_handler(websocket):
    """Handles a single client connection and its messages."""
    print("Client connected. Waiting for subscription...")
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "get_initial_state":
                    print("Received 'get_initial_state', sending all board states.")
                    await websocket.send(get_initial_state_message())
                    SUBSCRIBED_CLIENTS.add(websocket)
                    print(f"Client subscribed. Total subscribers: {len(SUBSCRIBED_CLIENTS)}")
                else:
                    print(f"Received unknown message type: {msg_type}")

            except json.JSONDecodeError:
                print("Received invalid JSON message.")
            except Exception as e:
                print(f"Error processing message: {e}")

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Client connection closed: {e.code} {e.reason}")
    finally:
        if websocket in SUBSCRIBED_CLIENTS:
            SUBSCRIBED_CLIENTS.remove(websocket)
            print(f"Client unsubscribed. Total subscribers: {len(SUBSCRIBED_CLIENTS)}")
        else:
            print("Disconnected an unsubscribed client.")


async def main():
    """Starts the WebSocket server and the move producer."""
    print(f"Starting WebSocket server on ws://{SERVER_HOST}:{SERVER_PORT}")
    server = websockets.serve(connection_handler, SERVER_HOST, SERVER_PORT)
    
    await asyncio.gather(
        server,
        move_producer()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer shutting down.")
