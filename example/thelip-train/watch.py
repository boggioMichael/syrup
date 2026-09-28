"""thelip-train's window on a Windows PC: it runs what is asked of it, one run
at a time, and waits for the next.

A run is asked for with a file in the work folder (C:\\Users\\Public\\thelip-server\\train):

    request.json    {"args": ["--languages", "he", "--hours", "10", ...]}

The arguments are run_all.py's and are checked by its own parser before
anything starts; nothing but run_all.py is ever run. A file named `stop` ends
the current run. The run's output shows in this window and goes to run.log;
status.json says what is running, since when, and how it ended.

Closing the window stops everything; asked again, a run continues where it
stopped (run_all.py keeps its progress in state.json).

    python watch.py [run_all.py arguments]    # the arguments: one run first, then wait
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.environ.get("THELIP_TRAIN_ROOT") or os.path.dirname(os.environ.get("THELIP_TRAIN_HOME") or os.path.join(HERE, "work"))
REQUEST = os.path.join(FOLDER, "request.json")
STOP = os.path.join(FOLDER, "stop")
STATUS = os.path.join(FOLDER, "status.json")
LOG = os.path.join(FOLDER, "run.log")
sys.path.insert(0, HERE)


def status(**fields) -> None:
    fields["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = STATUS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(fields, f, ensure_ascii=False, indent=1)
    os.replace(tmp, STATUS)


def valid(args) -> bool:
    """run_all.py's own parser decides; a list of strings only."""
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        return False
    from run_all import build_parser

    class Refused(Exception):
        pass

    def refuse(message=None, *_):
        raise Refused(message or "not a run")

    parser = build_parser()
    parser.error = refuse                                    # argparse would end this process instead
    parser.exit = lambda status=0, message=None: refuse(message)
    try:
        parser.parse_args(args)
        return True
    except Refused as e:
        print(f"refused {args}: {e}", flush=True)
        return False


def kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.kill()


def run(args) -> int:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"\n=== {started}: run_all.py {' '.join(args)}", flush=True)
    status(state="running", args=args, started=started)
    with open(LOG, "a", encoding="utf-8") as log:
        log.write(f"\n=== {started}: run_all.py {' '.join(args)}\n")
        env = dict(os.environ)
        if "--smoke" in args:   # a smoke run has a work folder of its own: the real run's progress stays untouched
            env["THELIP_TRAIN_HOME"] = (env.get("THELIP_TRAIN_HOME") or os.path.join(FOLDER, "work")).rstrip("\\/") + "-smoke"
        proc = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, "run_all.py"), *args], cwd=HERE, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        stopped = False
        import threading

        def pump():
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        while proc.poll() is None:
            if os.path.exists(STOP):
                os.remove(STOP)
                print("stop asked: ending the run", flush=True)
                kill_tree(proc)
                stopped = True
                break
            time.sleep(2)
        proc.wait()
        reader.join(timeout=5)
    code = proc.returncode
    ended = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    status(state="stopped" if stopped else ("done" if code == 0 else "failed"), args=args, started=started, ended=ended, exit=code)
    print(f"=== {ended}: {'stopped' if stopped else 'finished'} (exit {code}); waiting for the next run", flush=True)
    return code


def main() -> None:
    os.makedirs(FOLDER, exist_ok=True)
    first = sys.argv[1:]
    if first and valid(first):
        run(first)
    else:
        status(state="waiting")
    print(f"thelip-train is waiting for a run (a request.json in {FOLDER}); close this window to stop", flush=True)
    while True:
        if os.path.exists(REQUEST):
            try:
                with open(REQUEST, encoding="utf-8") as f:
                    args = json.load(f).get("args")
            except (OSError, ValueError, AttributeError) as e:
                args = None
                print(f"unreadable request: {e}", flush=True)
            try:
                os.replace(REQUEST, REQUEST + ".taken")
            except OSError:
                pass
            if args is not None and valid(args):
                run(args)
            else:
                status(state="waiting", refused=args)
        if os.path.exists(STOP):
            os.remove(STOP)
        time.sleep(3)


if __name__ == "__main__":
    main()
