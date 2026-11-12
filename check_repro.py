import subprocess, sys, json, glob

def snapshot():
    return {f: json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob("outputs/*.json"))}

def diff(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out += diff(a.get(k), b.get(k), f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f"{path}[{i}]")
    elif a != b:
        out.append((path, a, b))
    return out

snaps = []
for run in (1, 2):
    print(f"\n===== pipeline run {run} =====")
    subprocess.run([sys.executable, "run_pipeline.py"], check=True)
    snaps.append(snapshot())

diffs = []
for f in snaps[0]:
    diffs += [(f,) + d for d in diff(snaps[0][f], snaps[1].get(f))]

if diffs:
    print("\nDIFFERENCES FOUND:")
    for d in diffs:
        print(d)
else:
    print("\nIDENTICAL: results are reproducible")