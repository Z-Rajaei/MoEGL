"""Count lines of code (cloc, code lines only: blanks and comments excluded).

Targets
  moegl_core             : the reusable MoEGL implementation (paid once)   -> moegl/*.py
  moegl_config_exp1_*    : YAML configurations for the three Lopez datasets
  moegl_config_exp2/3    : YAML configurations for the Rahimi / Miranda encoders
  lopez                  : Lopez & Cuadrado bespoke encoder (Java core + filters
                           + loaders + the three dataset mains + json2graph.py)
  rahimi                 : Rahimi et al. bespoke encoder (netgan/encoder.py)
  miranda                : Miranda et al. bespoke encoder (to_graph.py,
                           encoders.py, metamodels.py)

Output: experiments/results/loc.json + experiments/results/loc_by_file.csv and a
Markdown table on stdout.
"""
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))       # Codes/MoEGL-v2/experiments
PKG = os.path.dirname(HERE)                             # Codes/MoEGL-v2
ROOT = os.path.dirname(os.path.dirname(PKG))            # project root
CLOC = os.path.join(ROOT, "Revision", "node_modules", ".bin",
                    "cloc.cmd" if os.name == "nt" else "cloc")

TARGETS = {
    "moegl_core": [os.path.join(PKG, "moegl")],
    "moegl_config_exp1_ecore": [os.path.join(PKG, "configs", "exp1_lopez_ecore.yaml")],
    "moegl_config_exp1_rds": [os.path.join(PKG, "configs", "exp1_lopez_rds.yaml")],
    "moegl_config_exp1_yakindu": [os.path.join(PKG, "configs", "exp1_lopez_yakindu.yaml")],
    "moegl_config_exp2": [os.path.join(PKG, "configs", "exp2_rahimi_carwash.yaml")],
    "moegl_config_exp3": [os.path.join(PKG, "configs", "exp3_miranda_fsm.yaml")],
    "lopez": [os.path.join(HERE, "bespoke_encoders", "lopez")],
    "rahimi": [os.path.join(HERE, "bespoke_encoders", "rahimi")],
    "miranda": [os.path.join(HERE, "bespoke_encoders", "miranda")],
}


def cloc(paths):
    cmd = [CLOC, "--json", "--quiet",
           "--exclude-dir=__pycache__,tests",
           "--not-match-f=__init__\\.py"] + [p.replace("\\", "/") for p in paths]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    per_lang = {k: v for k, v in data.items() if k not in ("header", "SUM")}
    return {
        "code": data["SUM"]["code"],
        "comment": data["SUM"]["comment"],
        "blank": data["SUM"]["blank"],
        "files": data["SUM"]["nFiles"],
        "languages": {k: v["code"] for k, v in per_lang.items()},
    }


def cloc_by_file(paths):
    cmd = [CLOC, "--json", "--quiet", "--by-file",
           "--exclude-dir=__pycache__,tests",
           "--not-match-f=__init__\\.py"] + [p.replace("\\", "/") for p in paths]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    rows = []
    for f, v in data.items():
        if f in ("header", "SUM"):
            continue
        rows.append({"file": f, "language": v.get("language"),
                     "code": v.get("code"), "comment": v.get("comment"),
                     "blank": v.get("blank")})
    return rows


def main():
    results = {}
    for key, paths in TARGETS.items():
        existing = [p for p in paths if os.path.exists(p)]
        if not existing:
            results[key] = {"error": "path missing", "paths": paths}
            continue
        results[key] = cloc(existing)
        results[key]["paths"] = [os.path.relpath(p, ROOT) for p in existing]

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "loc.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    rows = []
    for key, paths in TARGETS.items():
        existing = [p for p in paths if os.path.exists(p)]
        for r in cloc_by_file(existing):
            r["target"] = key
            rows.append(r)
    with open(os.path.join(HERE, "results", "loc_by_file.csv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target", "file", "language",
                                          "code", "comment", "blank"])
        w.writeheader()
        w.writerows(rows)

    print("| target | files | code | comment | blank | languages |")
    print("|---|---|---|---|---|---|")
    for key, r in results.items():
        if "error" in r:
            print(f"| {key} | - | {r['error']} | | | |")
        else:
            print(f"| {key} | {r['files']} | {r['code']} | {r['comment']} | {r['blank']} | {r['languages']} |")


if __name__ == "__main__":
    sys.exit(main())
