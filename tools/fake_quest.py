#!/usr/bin/env python3
"""Stand-in for the headset page: streams synthetic controller poses to teleop_server.py.
  .venv/bin/python tools/fake_quest.py [--hand right] [--seconds 6]
Presses B (align) first, squeezes after 1 s, moves the hand 6 cm forward (WebXR -z) and 4 cm up over 2 s, pulls the
trigger, releases. Behind a real motor loop, the arm really moves.
"""
import argparse
import asyncio
import json
import math
import ssl
import time

import aiohttp


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="wss://127.0.0.1:8443/ws")
    ap.add_argument("--hand", default="right", choices=("left", "right"))
    ap.add_argument("--seconds", type=float, default=6)
    ap.add_argument("--hz", type=float, default=90)
    args = ap.parse_args()
    ctx = ssl.create_default_context()
    ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
    async with aiohttp.ClientSession() as s, s.ws_connect(args.url, ssl=ctx) as ws:
        async def reader():
            async for m in ws:
                d = json.loads(m.data)
                if d.get("t") == "haptic":
                    print("haptic", d)
        rt = asyncio.ensure_future(reader())
        t0, seq = time.monotonic(), 0
        while (t := time.monotonic() - t0) < args.seconds:
            k = min(max((t - 1.5) / 2.0, 0), 1)
            k = (1 - math.cos(math.pi * k)) / 2
            p = [0.2, 1.0 + 0.04 * k, -0.3 - 0.06 * k]
            b = [0, 1 if 0.2 < t < 0.5 else 0, 0]  # B (align) is on the right controller; on the left it is Y = STOP
            H = {"ok": True, "p": p, "q": [0, 0, 0, 1], "aim": [0, 0, -1],
                 "sq": 1.0 if 1.0 < t < args.seconds - 1 else 0.0, "tr": k,
                 "b": b if args.hand == "right" else [0, 0, 0], "stick": [0, 0]}
            seq += 1
            hands = {args.hand: H}
            if args.hand != "right":
                hands["right"] = {**H, "sq": 0.0, "tr": 0.0, "b": b}
            await ws.send_str(json.dumps({"t": "xr", "seq": seq, "ts": t * 1000, "fps": args.hz, "head": None,
                                          "hands": hands}))
            await asyncio.sleep(1 / args.hz)
        rt.cancel()


asyncio.run(main())
