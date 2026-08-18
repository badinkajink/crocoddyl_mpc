"""Print the dt a cell's existing plans were solved at, or 0.02.

Exists so solve_tasks.sh does not have to embed a heredoc inside a heredoc,
and so the fallback lives in one place. 0.02 is this study's dt everywhere;
croco_run's own default is 0.01 and taking that would silently halve the
maneuver's duration.
"""
import glob
import json
import os
import sys

for f in sorted(glob.glob(os.path.join(sys.argv[1], "plan_*.json"))):
    try:
        print(json.load(open(f))["dt"])
        break
    except Exception:                                            # noqa: BLE001
        continue
else:
    print(0.02)
