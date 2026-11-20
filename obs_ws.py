#!/usr/bin/env python3
"""Minimal obs-websocket v5 client (run on the camera machine): sends requests to OBS without printing the password.

usage: python3 obs_ws.py '[{"requestType": "GetInputSettings", "requestData": {"inputName": "Media"}}]' [capture.jpg]
A GetSourceScreenshot request writes its image to the file given as the second argument.
"""
import asyncio
import base64
import hashlib
import json
import os
import sys
import time

import websockets

CONFIG = os.path.expanduser('~/.config/obs-studio/plugin_config/obs-websocket/config.json')


def auth_string(password: str, salt: str, challenge: str) -> str:
    secret = base64.b64encode(hashlib.sha256((password + salt).encode()).digest()).decode()
    return base64.b64encode(hashlib.sha256((secret + challenge).encode()).digest()).decode()


def hms(t: float) -> str:
    # same format as drawtext %{pts:hms} on an absolute timestamp: hours since 1970:MM:SS.mmm
    return f'{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:06.3f}'


async def main() -> None:
    conf = json.load(open(CONFIG))
    requests = json.loads(sys.argv[1])
    shot_path = sys.argv[2] if len(sys.argv) > 2 else None
    async with websockets.connect(f"ws://127.0.0.1:{conf.get('server_port', 4455)}", max_size=None) as ws:
        hello = json.loads(await ws.recv())
        identify = {'rpcVersion': 1, 'eventSubscriptions': 0}
        auth = hello['d'].get('authentication')
        if auth:
            identify['authentication'] = auth_string(conf['server_password'], auth['salt'], auth['challenge'])
        await ws.send(json.dumps({'op': 1, 'd': identify}))
        await ws.recv()
        for index, request in enumerate(requests):
            t0 = time.time()
            await ws.send(json.dumps({'op': 6, 'd': {'requestId': str(index), **request}}))
            response = json.loads(await ws.recv())['d']
            t1 = time.time()
            data = response.get('responseData') or {}
            if 'imageData' in data and shot_path:
                open(shot_path, 'wb').write(base64.b64decode(data['imageData'].split(',', 1)[1]))
                data = {'imageData': f'-> {shot_path}', 'obs_capture_time': f'{hms(t0)} .. {hms(t1)}'}
            print(json.dumps({'request': request['requestType'], 'status': response['requestStatus'], 'data': data},
                             ensure_ascii=False))


asyncio.run(main())
