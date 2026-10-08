#!/usr/bin/env python3
"""Per-node execution profiler for ComfyUI workflows.

Talks to a running ComfyUI server over HTTP + websocket, queues a workflow in
API prompt format, and records per-node wall-clock timings.

Modes:
  --extract OUT.json   Fetch the latest completed run from /history and save
                       its API-format prompt (run a workflow once from the GUI
                       first). Also prints run metadata.
  --run PROMPT.json    Queue the API prompt and record per-node timings.
                       Repeatable; for cold runs restart ComfyUI first.
  --list               List recent runs from /history.

Per-node end time is inferred: "executed" messages only fire for nodes that
produce UI output, otherwise a node is considered finished when the next node
starts executing. This is inherent to ComfyUI's protocol, not a limitation of
this script.

Requires ComfyUI's Python environment (uses aiohttp, which ships with it):
  E:\\AI\\ComfyUI_windows_portable\\python_embeded\\python.exe scripts/profile_workflow.py ...
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request


def http_json(url, payload=None, timeout=30):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_history(base, prompt_id=None):
    url = base + "/history" + (f"/{prompt_id}" if prompt_id else "")
    return http_json(url, timeout=60)


def api_prompt_of(entry):
    """Return the API-format prompt dict from a history entry.

    Queue items are stored as [number, prompt_id, prompt, extra_data,
    outputs_to_execute]; find the element that looks like an API prompt.
    """
    prompt = entry.get("prompt") or []
    for el in prompt:
        if isinstance(el, dict) and el and all(
            isinstance(v, dict) and "class_type" in v for v in el.values()
        ):
            return el
    return None


def pick_latest_run(history, require_nodes=None):
    """Return (prompt_id, entry) of the most recent completed run."""
    candidates = []
    for pid, entry in history.items():
        status = entry.get("status") or {}
        if not status.get("completed"):
            continue
        api_prompt = api_prompt_of(entry)
        if not api_prompt:
            continue
        if require_nodes:
            if not any(n.get("class_type") == require_nodes for n in api_prompt.values()):
                continue
        create_time = 0
        messages = status.get("messages") or []
        for ev, data in messages:
            if ev == "execution_start" and data.get("timestamp"):
                create_time = data["timestamp"]
        candidates.append((create_time, pid, entry, api_prompt))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    _, pid, entry, api_prompt = candidates[-1]
    return pid, entry, api_prompt


def cmd_extract(args):
    history = fetch_history(args.url)
    picked = pick_latest_run(history, require_nodes=args.require_node)
    if not picked:
        print("No completed run found in history" + (f" with node {args.require_node}" if args.require_node else ""))
        return 1
    pid, entry, api_prompt = picked
    with open(args.extract, "w", encoding="utf-8") as f:
        json.dump(api_prompt, f, indent=2, ensure_ascii=False)
    messages = entry.get("status", {}).get("messages", [])
    print(f"Saved API prompt from run {pid}")
    print(f"  nodes: {len(api_prompt)}")
    print(f"  status messages: {len(messages)}")
    for ev, data in messages:
        ts = data.get("timestamp")
        when = time.strftime("%H:%M:%S", time.localtime(ts / 1000)) if ts else "?"
        if ev in ("execution_start", "execution_success", "execution_error"):
            print(f"  [{when}] {ev}")
    print(f"  outputs saved for {len(entry.get('outputs', {}))} nodes")
    print(f"-> {args.extract}")
    return 0


def cmd_list(args):
    history = fetch_history(args.url)
    rows = []
    for pid, entry in history.items():
        status = entry.get("status") or {}
        messages = status.get("messages", [])
        t0 = t1 = None
        for ev, data in messages:
            ts = data.get("timestamp")
            if ev == "execution_start" and ts:
                t0 = ts
            if ev in ("execution_success", "execution_error") and ts:
                t1 = ts
        prompt = entry.get("prompt") or [None, None]
        api_prompt = api_prompt_of(entry) or {}
        duration = (t1 - t0) / 1000 if (t0 and t1) else None
        rows.append((t0 or 0, pid, status.get("status_str", "?"), duration, len(api_prompt)))
    rows.sort()
    print(f"{'start':19} {'prompt_id':38} {'status':8} {'dur_s':>8} {'nodes':>5}")
    for t0, pid, st, dur, n in rows[-args.n:]:
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0 / 1000)) if t0 else "?"
        print(f"{when:19} {pid:38} {st:8} {dur if dur is not None else float('nan'):8.1f} {n:5d}")
    return 0


class RunRecorder:
    def __init__(self, prompt_id, api_prompt):
        self.prompt_id = prompt_id
        self.api_prompt = api_prompt
        self.t0 = None  # monotonic base, set at execution_start/first message
        self.events = []  # (mono_t, kind, node_id)
        self.cached = set()
        self.error = None
        self.done = asyncio.get_running_loop().create_future()

    def now(self):
        if self.t0 is None:
            self.t0 = time.monotonic()
        return time.monotonic() - self.t0

    def handle(self, msg_type, data):
        if data.get("prompt_id") not in (None, self.prompt_id):
            return
        t = self.now()
        if msg_type == "execution_start":
            self.events.append((t, "start", None))
        elif msg_type == "execution_cached":
            for nid in data.get("nodes", []):
                self.cached.add(str(nid))
            self.events.append((t, "cached", ",".join(str(n) for n in data.get("nodes", []))))
        elif msg_type == "executing":
            nid = data.get("node")
            if nid is not None:
                self.events.append((t, "executing", str(nid)))
        elif msg_type == "executed":
            nid = data.get("node")
            if nid is not None:
                self.events.append((t, "executed", str(nid)))
        elif msg_type == "execution_success":
            self.events.append((t, "success", None))
            if not self.done.done():
                self.done.set_result(True)
        elif msg_type == "execution_error":
            self.error = data.get("exception_message") or data.get("node_type") or "error"
            self.events.append((t, "error", str(data.get("node_id"))))
            if not self.done.done():
                self.done.set_result(False)

    def compute_rows(self):
        """Derive per-node durations from the event sequence."""
        rows = []
        open_node = None
        open_start = 0.0
        total = self.events[-1][0] if self.events else 0.0
        for t, kind, nid in self.events:
            if kind == "executing":
                if open_node is not None:
                    rows.append(self._row(open_node, open_start, t - open_start, "next"))
                open_node = nid
                open_start = t
            elif kind == "executed":
                if open_node == nid:
                    rows.append(self._row(nid, open_start, t - open_start, "executed"))
                    open_node = None
                else:
                    rows.append(self._row(nid, t, 0.0, "executed-without-start"))
            elif kind in ("success", "error"):
                if open_node is not None:
                    rows.append(self._row(open_node, open_start, t - open_start, kind))
                    open_node = None
                total = t
        return rows, total

    def _row(self, nid, start, dur, end_source):
        node = self.api_prompt.get(nid, {})
        return {
            "node_id": nid,
            "class_type": node.get("class_type", "?"),
            "start_s": round(start, 3),
            "duration_s": round(dur, 3),
            "end_source": end_source,
            "cached": nid in self.cached,
        }


async def run_once(base, api_prompt, args):
    import aiohttp

    client_id = os.urandom(16).hex()
    ws_url = base.replace("http://", "ws://").replace("https://", "wss://") + f"/ws?clientId={client_id}"

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(ws_url, heartbeat=30) as ws:
            resp = await session.post(
                base + "/prompt",
                json={"prompt": api_prompt, "client_id": client_id},
                timeout=aiohttp.ClientTimeout(total=60),
            )
            body = await resp.json()
            if resp.status != 200:
                raise SystemExit(f"POST /prompt failed ({resp.status}): {json.dumps(body)[:2000]}")
            prompt_id = body["prompt_id"]
            print(f"Queued prompt {prompt_id}")

            recorder = RunRecorder(prompt_id, api_prompt)

            try:
                async with asyncio.timeout(args.timeout):
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            recorder.handle(data.get("type"), data.get("data") or {})
                            if recorder.done.done():
                                break
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
            except TimeoutError:
                raise SystemExit(f"Timeout after {args.timeout}s waiting for execution to finish")

            if not recorder.done.done():
                raise SystemExit("Websocket closed before execution finished")

            rows, total = recorder.compute_rows()
            return recorder, rows, total


def print_table(rows, total, top):
    executed = [r for r in rows if not r["cached"] and r["duration_s"] > 0]
    print(f"\nTotal: {total:.1f}s   nodes executed: {len(executed)}   cached: {sum(r['cached'] for r in rows)}")
    if total <= 0:
        return
    order = sorted(executed, key=lambda r: r["duration_s"], reverse=True)
    if top > 0:
        order = order[:top]
    print(f"\n{'node':>6} {'time_s':>8} {'%':>6}  {'start_s':>8}  class_type")
    for r in order:
        pct = 100.0 * r["duration_s"] / total if total else 0
        print(f"{r['node_id']:>6} {r['duration_s']:8.1f} {pct:5.1f}%  {r['start_s']:8.1f}  {r['class_type']}  ({r['end_source']})")


def write_csv(path, rows, total):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["node_id", "class_type", "start_s", "duration_s", "pct_total", "end_source", "cached"])
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["start_s"]):
            out = dict(r)
            out["pct_total"] = round(100.0 * r["duration_s"] / total, 2) if total else 0
            w.writerow(out)
    print(f"CSV: {path}")


def cmd_run(args):
    with open(args.run, encoding="utf-8") as f:
        api_prompt = json.load(f)

    recorder, rows, total = asyncio.run(run_once(args.url, api_prompt, args))
    if recorder.error:
        print(f"Run finished with ERROR: {recorder.error}")
    print_table(rows, total, args.top)
    if args.csv:
        write_csv(args.csv, rows, total)
    return 1 if recorder.error else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8188", help="ComfyUI base URL")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--extract", metavar="OUT.json", help="Save latest run's API prompt from /history")
    mode.add_argument("--run", metavar="PROMPT.json", help="Queue an API prompt and record per-node timings")
    mode.add_argument("--list", action="store_true", help="List recent runs from /history")
    p.add_argument("-n", type=int, default=20, help="With --list: how many runs to show")
    p.add_argument("--require-node", default="VaeDecodeStructureTrellis2",
                   help="With --extract: only accept runs containing this node class (default: %(default)s)")
    p.add_argument("--csv", help="With --run: also write timings CSV to this path")
    p.add_argument("--top", type=int, default=25, help="With --run: rows in the summary table (0 = all)")
    p.add_argument("--timeout", type=float, default=1800, help="With --run: max seconds to wait")
    args = p.parse_args()

    if args.extract:
        return cmd_extract(args)
    if args.list:
        return cmd_list(args)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
