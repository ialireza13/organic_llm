"""Tiny job queue: run the commands in a jobs file with at most N in parallel.

Jobs file: one job per line, "name<TAB>command". A job is skipped if runs/queue/<name>.done exists,
so the queue can be restarted safely. The jobs file is re-read whenever a slot frees up, so jobs can
be appended while the queue runs. Stops launching new jobs after --deadline (unix time).
"""
import argparse
import os
import subprocess
import time


def read_jobs(path):
    jobs = []
    for line in open(path):
        line = line.rstrip("\n")
        if not line.strip() or line.startswith("#"):
            continue
        name, cmd = line.split("\t", 1)
        jobs.append((name, cmd))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--max_parallel", type=int, default=4)
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--state", default="runs/queue")
    a = ap.parse_args()
    os.makedirs(a.state, exist_ok=True)
    running = {}
    started = set()
    while True:
        for name, p in list(running.items()):
            rc = p.poll()
            if rc is not None:
                with open(os.path.join(a.state, f"{name}.{'done' if rc == 0 else 'failed'}"), "w") as f:
                    f.write(f"rc={rc} t={time.time()}\n")
                print(f"{time.strftime('%H:%M:%S')} finished {name} rc={rc}", flush=True)
                del running[name]
        pending = [(n, c) for n, c in read_jobs(a.jobs)
                   if n not in started and not os.path.exists(os.path.join(a.state, f"{n}.done"))]
        if a.deadline and time.time() > a.deadline:
            pending = []
        while pending and len(running) < a.max_parallel:
            name, cmd = pending.pop(0)
            log = open(os.path.join("runs", f"{name}.log"), "w")
            running[name] = subprocess.Popen(cmd, shell=True, executable="/bin/bash", stdout=log, stderr=subprocess.STDOUT)
            started.add(name)
            print(f"{time.strftime('%H:%M:%S')} started {name}: {cmd}", flush=True)
        if not running and not pending:
            print("QUEUE_EMPTY", flush=True)
            break
        time.sleep(10)


if __name__ == "__main__":
    main()
