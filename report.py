#!/usr/bin/env python
"""
extract reports from run directory of benchmarks

"""
from pathlib import Path
import logging as log
import json
import sys
import argparse
import numpy as np
from tqdm.auto import tqdm

# Blob dir
RUN_DIR = Path("/mnt/tg/data/papers/2024-pymarian/benchmarks/wmt23-v2")
# or local dir; TODO: update this path
RUN_DIR = Path("wmt23")
log.basicConfig(level=log.INFO)

TABLE_FORMATS = ['human', 'latex']

def display_name(name):
    fix = {
        'marianbin': 'Marian',
        'orig': 'Orig',
        'pymarian': 'PyM',
        'pymarian_fp16': 'PyM FP16',
        '1gpus': '1',
        '8gpus': '8',
    }
    return fix.get(name, name)


def scan_files(path: Path=RUN_DIR):
    prefix = 'all.scores'
    suffix = 'stats.json'
    stats_files = list(path.glob(f"{prefix}.*.{suffix}"))
    assert stats_files, f"No stats files found in {path}/{prefix}.*.{suffix}"
    metrics = []
    impls = []
    gpus = []
    lookup = {}   # file to (metric, impl, gpus)
    for f in stats_files:
        name = f.name.replace(f"{prefix}.", "").replace(f".{suffix}", "")
        parts = name.split('.')
        if len(parts) == 4 and parts[-1] == 'fp16':
            # initially stored as metric.pymarian.ngpus.fp16, then renamed to <metric>.pymarian_fp16.<n>gpus
            parts[1]+= '_fp16'
            del parts[-1]
        assert len(parts) == 3, f"Invalid name {name}"
        assert parts[-1].endswith('gpus'), f"Invalid name {name}"
        if parts[0].startswith("wmt21"):
            continue   # skip wmt21
        tag = tuple(parts)
        if tag in lookup:
            log.warning(f"Duplicate tag {tag} found in {f}. prev: {lookup[tag]}, current: {f}")
        lookup[tag] = f

    all_tags = set(lookup.keys())
    metrics = list(sorted(set(t[0] for t in all_tags)))
    impls = ['orig', 'marianbin', 'pymarian', 'pymarian_fp16']
    gpus = ['1gpus', '8gpus']
     # any new unexpected tags
    impls_found = list(set(t[1] for t in all_tags))
    impls += [x for x in impls_found if x not in impls]
    gpus_found = list(set(t[2] for t in all_tags))
    gpus += [x for x in gpus_found if x not in gpus]
    return dict(
        metrics=metrics,
        impls=impls,
        gpus=gpus,
        lookup=lookup
    )


def parse_scores(metrics, impls, gpus, lookup) -> dict:
    # expected table
    def parse_scores(path:Path) -> np.ndarray:
        assert path.exists(), f"File {path} not found"
        assert path.stat().st_size > 0, f"File {path} is empty"
        if path.name.endswith('.json'):  # comet-score
            data = json.loads(path.read_text())
            # exactly one key at the top level
            assert len(data) == 1, f"Invalid comet-score file {path}"
            scores = list(data.values())[0]
            scores = np.array([x['COMET'] for x in scores])
        else:
            scores = np.loadtxt(path)
        return scores

    #impls = ['orig', 'marianbin' ,'pymarian_fp16']
    bar = tqdm(desc='Reading scores', total=len(metrics) * len(impls) * len(gpus))

    cache = {}
    for g in gpus:
        for m in metrics:
            for i in impls:
                p = lookup.get((m, i, g), None)
                if p:
                    ext = 'txt'
                    if i == 'orig' and m != 'bleurt-20':   # comet-score is stored as json
                        ext = 'json'
                    glob = f"{p.name.replace('.stats.json', '')}.*.{ext}"
                    score_files = list(p.parent.glob(glob))
                    score_files = [f for f in score_files if f.name != p.name and f.stat().st_size > 0]
                    if score_files:
                        scores = []
                        for f in score_files:
                            score = parse_scores(f).mean()
                            scores.append(score)
                            bar.set_postfix_str(f"{g} {m} {i} :: {f.name} :: {score:.4f}")

                        scores = [round(x, 6) for x in scores]
                        if len(set(scores)) == 1: # all runs yield same score. good
                            cache[(m, i, g)] =  scores[0]
                        else:
                            log.warning(f"Scores are not deterministic across runs for {m} {i} {g}: {scores}")
                        if len(scores) != 3:
                            log.warning(f"Expected 3 scores, got {len(scores)} for {m} {i} {g}: {scores}")
                else:
                    log.warning(f"Missing score files for {m} {i} {g}")
                bar.update(1)
    return cache

def parse_timings(metrics, impls, gpus, lookup, missing='NA'):

    # expected table
    """
    Metric   1gpu 1gpu ... 8gpus 8gpus ...
    Metric   orig marian ... orig marian ...
    Bleurt    ?   ? .... ? ?
    ....
    """
    # mean runtime
    assert impls[0] == 'orig', "orig should be the first impl"
    rows = []
    rows.append(['', 'Time (seconds)'] + [''] *(len(impls) -1) + ['Speed Ratio'] + [''] *(len(impls) -2))
    rows.append(['Metric'] + [display_name(x) for x in impls] + [display_name(x) for x in impls[1:]])

    cache = {}
    for g in gpus:
        rows.append([f"{g} GPUs"])
        for m in metrics:
            rows.append([m])
            #rows.append([m, display_name(g)])
            for i in impls:
                p = lookup.get((m, i, g), None)
                val = missing
                if p and p.exists() and p.stat().st_size > 0:
                    data = json.loads(p.read_text())
                    res = data['results'][0]
                    val = f"{int(res['mean'])}~{res['stddev']:.1f}"
                    cache[(m, i, g)] = res['mean']
                rows[-1].append(val)
            for i in impls[1:]:
                val = missing
                if (m, i, g) in cache and (m, 'orig', g) in cache:
                    ratio =  cache[(m, 'orig', g)] / cache[(m, i, g)]
                    val = f"{ratio:.1f}x"
                rows[-1].append(val)
    return rows

def parse_score_diffs(metrics, impls, gpus, lookup, missing='NA'):
    scores = parse_scores(metrics, impls, gpus, lookup)

    assert 'orig' == impls[0], "orig should be the first impl"
    rows = []
    rows.append(['Metric'] + ['Score'] * len(impls) + ['Error'] * len(impls[1:]))
    rows.append([''] + [display_name(x) for x in impls] + [display_name(x) for x in impls[1:]])

    g = gpus[0]   # only one gpu
    for m in metrics:
        rows.append([m])
        for i in impls:
            val = missing
            if (m, i, g) in scores:
                val = f"{scores[(m, i, g)]:.4f}"
            rows[-1].append(val)

        #extend the row with error margins
        for i in impls[1:]:
            err = missing
            if (m, i, g) in scores and (m, 'orig', g) in scores:
                err =  abs(scores[(m, i, g)] - scores[(m, 'orig', g)])
                err = f"{err:.4f}"
            rows[-1].append(err)
    return rows


def print_table(table, format=TABLE_FORMATS[0], out=sys.stdout):
    assert format in TABLE_FORMATS, f"Invalid format {format}"
    n_cols = max(len(r) for r in table)
    if format == 'human':
        delim = ' | '
        end = '\n'
        col_widths = [max(len(str(r[i])) if len(r) > i else 0 for r in table) for i in range(n_cols)]
        for r in table:
            pad_r = [f"{c:<{w}}" for c, w in zip(r, col_widths)]
            out.write(delim.join(pad_r) + end)
    elif format == 'latex':

        delim = ' & '
        end = ' \\\\\n'
        header = ''''
\\begin{table*}[htb]
\\centering
\\begin{tabular}{ l ''' + '| r ' * (len(table[0]) -1) + '''}
\\toprule
'''
        footer = '''\\bottomrule
\\end{tabular}
\\label{tab:id}
\\caption{Caption}
\\end{table*}
'''
        out.write(header)
        for r in table:
            r = [c.replace('~', r'$\pm$').replace('_', r'\_').replace('%', r'\%') for c in r]
            out.write(delim.join(r) + end)
        out.write(footer)
    else:
        raise ValueError(f"Invalid or unsupported format {format}")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--dir", type=Path, default=RUN_DIR, help="run directory")
    parser.add_argument("-f", "--format", type=str, choices=TABLE_FORMATS,
                        default=TABLE_FORMATS[0],
                        help="output format")
    #parser.add_argument('--task', '-t', type=str, choices=['times', 'errmargin'], default='times', help='task to report')

    args = parser.parse_args()
    meta = scan_files()
    rows = parse_timings(**meta, )
    print_table(rows, format=args.format)
    if True:
        print("\n\n")
        meta['impls'] = ['orig', 'pymarian', 'pymarian_fp16']
        rows = parse_score_diffs(**meta)
        print_table(rows, format=args.format)

if __name__ == '__main__':
    main()


# update files on blob
# d=/mnt/tg/data/papers/2024-pymarian/benchmarks/wmt23-v2; ls wmt23/all.scores.* | grep -v '\.bak' | while read i; do [[ -s $d/$(basename $i) ]] && continue; echo cp $i $d/; done
