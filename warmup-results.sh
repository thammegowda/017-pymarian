#!/usr/bin/env bash
# this script for generating warmup report
set -eu 
# print the results of the loading benchmarks
delim='\t'
delim=','
pref="/mnt/tg/data/papers/2024-pymarian/benchmarks/wmt23/loader"
header1="Model${delim}Time 1GPU${delim}${delim}Time 8GPUs${delim}${delim}Memory (MB) 1GPU${delim}${delim}Memory (MB) 8GPU${delim}"
header2="Model${delim}Orig${delim}Pymarian${delim}Speedup${delim}${delim}Orig${delim}Pymarian${delim}Speedup${delim}Orig${delim}Pymarian${delim}Orig${delim}Pymarian"
echo -e "$header1"
echo -e "$header2"

metrics=($(
    ls $pref.*.1gpus.stats.json | xargs -n1 basename  | cut -f2 -d. | sort -u
))
for m in ${metrics[@]}; do
    echo -en "$m"
    key=".time.mean"
    for n_gpus in 1 8; do
        for i in orig pymarian; do #pymarian.fp16
            out=$pref.$m.$i.${n_gpus}gpus.stats.json
            [[ -s $out ]] && val=$(jq -r "${key}" $out) || val="NA"
            echo -en "${delim}$val"
        done
        echo -en "${delim}?"
    done
    # memory
    key=".mem.max"
    for n_gpus in 1 8; do
        for i in orig pymarian; do #pymarian.fp16
            out=$pref.$m.$i.${n_gpus}gpus.stats.json
            [[ -s $out ]] && val=$(jq -r "${key}" $out) || val="NA"
            echo -en "${delim}$val"
        done
    done
    echo ""   # new line
done

