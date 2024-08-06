# measure the max memory taken by a program given by cli args
# usage: python memorybench.py "cmdline"
"""
python memorybech.py "python -c 'import numpy as np; a=np.arange(1000000000, dtype=np.int8)'"
python memorybech.py "python -c 'import numpy as np; a=np.arange(1000000000, dtype=np.int16)'"
python memorybech.py "python -c 'import numpy as np; a=np.arange(1000000000, dtype=np.int32)'"
python memorybech.py "python -c 'import numpy as np; a=np.arange(1000000000, dtype=np.int64)'"
"""

import argparse
import sys
import resource
import subprocess
import json
import numpy as np

parser = argparse.ArgumentParser(
    description="Measure the max memory taken by a program given by cli args",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("cmd", type=str, help="command to run")
parser.add_argument("-n", "--runs", type=int, default=3, help="number of runs")
parser.add_argument(
    "-o",
    "--output",
    type=argparse.FileType("w"),
    default=sys.stdout,
    help="output file to write stats",
)
args = parser.parse_args()


mems = []
times = []
exitcodes = []
for i in range(args.runs):
    print(f"Run {i+1}/{args.runs}", file=sys.stderr)
    proc = subprocess.Popen(args.cmd, shell=True)
    proc.wait()
    exitcodes.append(proc.returncode)
    chres = resource.getrusage(resource.RUSAGE_CHILDREN)
    mem_mb = round(chres.ru_maxrss / 1024, 3)
    tot_time = round(chres.ru_utime + chres.ru_stime, 3)
    mems.append(mem_mb)
    if not times:
        times.append(tot_time)
    else:
        # calculate the time taken by the current run
        times.append(round(tot_time - sum(times), 3))

# note:we get MAX memory usage of the child process.
# once a run has reached its max memory usage, all subsequent runs will have the same max memory usage
# so, mean and min for memory usage is not useful
stats = {
    "cmd": args.cmd,
    "runs": args.runs,
    "mem": {
        "max": max(mems),
        # "mean": round(sum(mems) / len(mems), 3),
        # "std": round(np.std(mems), 3),
        "runs": mems,
        # "min": min(mems),
        "unit": "MB",
    },
    "time": {
        "mean": round(sum(times) / len(times), 3),
        "std": round(np.std(times), 3),
        "min": min(times),
        "max": max(times),
        "runs": times,
        "unit": "s",
    },
    'exitcode': exitcodes,
}


args.output.write(json.dumps(stats, indent=2))
