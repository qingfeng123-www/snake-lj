import argparse
import asyncio
import json
from typing import Dict

import websockets
from websockets.server import WebSocketServerProtocol


CLIENTS: Dict[str, WebSocketServerProtocol] = {}


async def send_json(ws: WebSocketServerProtocol, payload: dict) -> None:
    await ws.send(json.dumps(payload, ensure_ascii=False))


async def forward_to_target(sender_id: str, payload: dict) -> None:
    target_id = payload.get("to", "")
    if not target_id:
        ws = CLIENTS.get(sender_id)
        if ws:
            await send_json(ws, {
                "type": "relay_error",
                "message": "missing target id"
            })
        return

    target_ws = CLIENTS.get(target_id)
    if not target_ws:
        ws = CLIENTS.get(sender_id)
        if ws:
            await send_json(ws, {
                "type": "relay_error",
                "to": target_id,
                "message": "target offline"
            })
        return

    await send_json(target_ws, payload)


async def handler(ws: WebSocketServerProtocol) -> None:
    client_id = ""
    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await send_json(ws, {
                    "type": "relay_error",
                    "message": "invalid json"
                })
                continue

            msg_type = msg.get("type", "")

            if msg_type == "register":
                proposed = str(msg.get("clientId", "")).strip()
                if not proposed:
                    await send_json(ws, {
                        "type": "relay_error",
                        "message": "missing clientId"
                    })
                    continue

                if proposed in CLIENTS and CLIENTS[proposed] is not ws:
                    await send_json(ws, {
                        "type": "relay_error",
                        "message": "clientId already in use"
                    })
                    continue

                client_id = proposed
                CLIENTS[client_id] = ws
                await send_json(ws, {
                    "type": "register_ack",
                    "clientId": client_id
                })
                continue

            if not client_id:
                await send_json(ws, {
                    "type": "relay_error",
                    "message": "register first"
                })
                continue

            if msg_type in {
                "relay_offer",
                "relay_accept",
                "relay_reject",
                "relay_payload",
                "relay_disconnect"
            }:
                msg["from"] = client_id
                await forward_to_target(client_id, msg)
                continue

            await send_json(ws, {
                "type": "relay_error",
                "message": "unknown message type"
            })

    finally:
        if client_id and CLIENTS.get(client_id) is ws:
            del CLIENTS[client_id]


async def main() -> None:
    parser = argparse.ArgumentParser(description="WebSocket relay server for snake game fallback mode.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    async with websockets.serve(handler, args.host, args.port):
        print(f"Relay server listening on ws://{args.host}:{args.port}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
