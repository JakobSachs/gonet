import asyncio, json
import websockets

clients = set()

async def handler(ws, _path):
    clients.add(ws)
    try:
        await ws.wait_closed()
    finally:
        clients.remove(ws)

async def broadcaster(event_queue):
    loop = asyncio.get_event_loop()
    while True:
        msg = await loop.run_in_executor(None, event_queue.get)
        data = json.dumps(msg)
        if clients:
            await asyncio.wait([c.send(data) for c in clients])

def start_ws(host, port, event_queue):
    loop = asyncio.get_event_loop()
    loop.create_task(broadcaster(event_queue))
    ws_srv = websockets.serve(handler, host, port)
    loop.run_until_complete(ws_srv)
    loop.run_forever()
