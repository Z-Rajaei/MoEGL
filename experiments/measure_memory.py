"""Peak memory of the Table 2 encoders, same commands as the timing protocol.

Does not write exp_results.json.

For each of N fresh processes (Java TimeLopezAll on the 500 Ecore models,
then MoEGL time_moegl.py on exp1_lopez_ecore.yaml) the Windows process
counters are read after the process exits:

  PeakWorkingSetSize  physical RAM high-water mark
  PeakPagefileUsage   committed-memory high-water mark

Usage: python experiments/measure_memory.py [--runs 10]
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PKG))
TCRMG = "E:/Project/TCRMG-GNN-1.0.0"
JAVA_BIN = "C:/Program Files/Android/Android Studio/jbr/bin/java.exe"
LOPEZ_CP = ";".join([
    os.path.join(HERE, "lopez_driver", "classes"),
    os.path.join(ROOT, "Revision", "lopez_timing", "lib", "*"),
    f"{TCRMG}/java/GraphGeneration/target/classes",
])
CONFIG = os.path.join(PKG, "configs", "exp1_lopez_ecore.yaml")
TIME_MOEGL = os.path.join(HERE, "time_moegl.py")
OUT = os.path.join(HERE, "results", "memory_paired_ecore.json")


def snapshot_tree(root_pid, seen):
    """Record PeakWorkingSet and PeakPagefile for the process and its descendants.

    The venv python.exe is a launcher; the interpreter that holds the graphs
    is a child. Peak counters are cumulative, so the maximum observed while
    the tree is alive is the high-water mark.
    """
    try:
        root = psutil.Process(root_pid)
        procs = [root] + root.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for p in procs:
        try:
            mi = p.memory_info()
            rec = seen.setdefault(p.pid, {
                "name": p.name(),
                "peak_working_set": 0,
                "peak_pagefile": 0,
            })
            rec["name"] = p.name()
            rec["peak_working_set"] = max(rec["peak_working_set"], int(mi.peak_wset))
            rec["peak_pagefile"] = max(rec["peak_pagefile"], int(mi.peak_pagefile))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def run_once(cmd):
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    seen = {}
    while proc.poll() is None:
        snapshot_tree(proc.pid, seen)
        time.sleep(0.02)
    snapshot_tree(proc.pid, seen)
    out, err = proc.communicate()
    wall_ms = (time.perf_counter() - t0) * 1000
    if proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode}\n{out}\n{err}")
    if not seen:
        raise RuntimeError("no memory sample; process tree was not observed")
    worker = max(seen.values(), key=lambda r: r["peak_working_set"])
    return {
        "peak_working_set": worker["peak_working_set"],
        "peak_pagefile": worker["peak_pagefile"],
        "worker_name": worker["name"],
        "tree": [{"pid": pid, **rec} for pid, rec in seen.items()],
        "process_ms": wall_ms,
        "stdout": out,
    }


def mean_sd(values):
    return {
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
        "values": values,
    }


def summarize(rows, key):
    return mean_sd([r[key] for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()
    java_cmd = [
        JAVA_BIN, "-cp", LOPEZ_CP, "TimeLopezAll",
        "ecore", f"{TCRMG}/realModels/Ecore", "-", "-",
    ]
    py_cmd = [sys.executable, TIME_MOEGL, CONFIG]
    java_rows, py_rows = [], []
    for i in range(args.runs):
        row = run_once(java_cmd)
        java_rows.append(row)
        print(f"java {i+1}/{args.runs} peak_wset={row['peak_working_set']} "
              f"peak_commit={row['peak_pagefile']} ms={row['process_ms']:.0f}",
              flush=True)
    for i in range(args.runs):
        row = run_once(py_cmd)
        py_rows.append(row)
        print(f"moegl {i+1}/{args.runs} peak_wset={row['peak_working_set']} "
              f"peak_commit={row['peak_pagefile']} ms={row['process_ms']:.0f}",
              flush=True)
        print(row["stdout"].strip(), flush=True)
    result = {
        "protocol": (
            "N fresh processes, all Java then all MoEGL, Ecore 500. "
            "Same commands as time_lopez (TimeLopezAll ecore, no JSON output) and "
            "time_moegl.py on exp1_lopez_ecore.yaml. Every 20 ms the process tree "
            "is sampled with psutil (PeakWorkingSetSize and PeakPagefileUsage). "
            "The reported peak is the maximum of the tree, which is the worker "
            "process: java.exe, or the base python.exe spawned by the venv launcher."
        ),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "n": args.runs,
        "lopez_java": {
            "peak_working_set_bytes": summarize(java_rows, "peak_working_set"),
            "peak_pagefile_bytes": summarize(java_rows, "peak_pagefile"),
            "process_ms": summarize(java_rows, "process_ms"),
            "worker_names": [r["worker_name"] for r in java_rows],
        },
        "moegl": {
            "peak_working_set_bytes": summarize(py_rows, "peak_working_set"),
            "peak_pagefile_bytes": summarize(py_rows, "peak_pagefile"),
            "process_ms": summarize(py_rows, "process_ms"),
            "worker_names": [r["worker_name"] for r in py_rows],
            "last_stdout": py_rows[-1]["stdout"],
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
