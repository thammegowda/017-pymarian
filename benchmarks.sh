#!/usr/bin/env bash
MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

BLOB_ROOT="/mnt/tg/data/papers/2024-pymarian/benchmarks/"  # blob storage mounted here
ROOT=$MYDIR
#[[ -d $BLOB_ROOT ]] && ROOT=$BLOB_ROOT

TESTSET="wmt23"
DATA="$ROOT/$TESTSET/all.tsv"
N_GPUS=8
GPU_IDS="$(seq 0 $((N_GPUS-1)) | tr '\n' ' ')"
BATCH_SIZE=128
RUN_WARMUP=1
export NCCL_DEBUG=WARN


MAX_SEGS= # empty for full data
N_RUNS=3

# if -q/--quick is passed, we will use a subset of data
if [[ "$1" == "--quick" || "$1" == "-q" ]]; then
    MAX_SEGS=100         # FIXME: this is for testing; comment out for the full run
    N_RUNS=2
    echo "Running in quick mode: using $MAX_SEGS segments and $N_RUNS runs" >&2
else
    echo "Running in full mode: using all segments and $N_RUNS runs" >&2
fi

TOOLS=(
    sacrebleu
    pymarian-eval
    comet-score
    hyperfine   # for benchmarking,  https://github.com/sharkdp/hyperfine
    wget
    unzip
)
METRICS=(
    # these are the metrics we are benchmarking
    bleurt-20
    wmt22-comet-da
    wmt22-cometkiwi-da
    wmt23-cometkiwi-da-xl
    wmt20-comet-qe-da
    wmt20-comet-qe-da-v2
    wmt20-comet-da
    #wmt21-comet-da
    #wmt21-comet-qe-da
    #wmt21-comet-qe-mqm
    # the below metrics are supported but we are not benchmarking them
    #wmt23-cometkiwi-da-xxl
)

log(){
    echo -e "$(date -Is) - $@" >&2
}

check_tools(){
    for tool in "${TOOLS[@]}"; do
        if ! command -v $tool &> /dev/null; then
            log "Error: $tool is not installed or missing in PATH"
            if [[ "$tool" == "hyperfine" ]]; then
                log "https://github.com/sharkdp/hyperfine. On ubuntu, you may try 'sudo apt install hyperfine'"
            fi
            exit 1
        fi
    done
    # if bleurt-20 in METRICS, check bleurt installation
    if [[ " ${METRICS[@]} " =~ " bleurt-20 " ]]; then
        if [[ $(python -m pip list | grep BLEURT | wc -l) -le 0 ]]; then
            log "Error: bleurt is not installed or missing in PATH"
            log "pip install git+https://github.com/google-research/bleurt.git"
            exit 1
        fi
        if [[ ! -d $ROOT/BLEURT-20 ]]; then
            log "Downloading BLEURT-20"
            wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip -O $ROOT/BLEURT-20.zip.tmp && mv $ROOT/BLEURT-20.zip{.tmp,}
            unzip $ROOT/BLEURT-20.zip -d $ROOT/
        fi
    fi
}

get_data(){

    if [ -s $DATA ]; then
        log "Data already exists at $DATA"
        return
    fi
    log "scanning the list of languages and system names for $TESTSET"
    mkdir -p $(dirname $DATA)
    rm -f $DATA  # remove empty file
    declare -A NAMES
    while IFS='\n' read line; do
        line=$(echo $line | sed 's/\r//g; s/,//g; s/: / /g')
        IFS=' ' read -r -a array <<< "$line"
        pair="${array[0]}"
        NAMES[$pair]=""
        skips=($pair src docid origlang domain)
        for it in "${array[@]}"; do
            for i in "${skips[@]}"; do
                if [[ "${i}" == "$it" ]]; then  #|| "$it" == ref*
                    #log "Skipping $it"
                    continue 2
                fi
            done
            NAMES[$pair]+="${it} "
        done
    done < <(sacrebleu -t $TESTSET --list)

    log "Downloading data for $TESTSET -> $DATA"
    for pair in "${!NAMES[@]}"; do
        for sysname in ${NAMES[$pair]}; do
            log "Downloading $pair $sysname"
            sacrebleu -t $TESTSET -l $pair --echo src ref $sysname | sed "s/^/$pair\t$sysname\t/"
        done
    done > $DATA.tmp && mv $DATA.tmp $DATA
}

measure_loading(){
    # measure the memory and time it takes to load the model with a small input (1 sentence)
    pref=$(dirname $DATA)/loader
    inp_txt=$pref.txt
    [[ -f $inp_txt ]] || echo "Hello" >  $inp_txt
    local data_args="-s $inp_txt -r $inp_txt -t $inp_txt"
    local n_gpus=1

    runner="python $MYDIR/memorybench.py -n 3"
    for n_gpus in 8 1; do
        local gpu_ids="$(seq 0 $((n_gpus-1)) | tr '\n' ' ')"
        export CUDA_VISIBLE_DEVICES="$(seq 0 $((n_gpus-1)) | tr '\n' ',' | sed 's/,$//')"

        for m in ${METRICS[@]}; do
            # Pymarian
            out=$pref.$m.pymarian.${n_gpus}gpus.stats.json
            cmd="pymarian-eval --devices $gpu_ids -m $m $data_args --mini-batch 1"
            [[ -s $out ]] || {
                $runner "$cmd" -o $out
            }
            out="${out%.stats.json}.fp16.stats.json"
            [[ -s $out ]] || {
                $runner "$cmd --fp16" -o $out
            }

            # the c++ version; get cmd from pymarian --print-cmd
            cmd=$(pymarian-eval --print-cmd --devices $gpu_ids -m $m $data_args --mini-batch $BATCH_SIZE)
            out="$pref.$m.marianbin.${n_gpus}gpus.stats.json"
            [[ -f $out ]] || {
                log "Benchmarking $m : marian-cpp; out=$out"
                #we need to decide input fields based on the model --like
                input="$inp_txt $inp_txt"
                if [[ "$cmd " =~ "--like comet " ]]; then  # three inputs
                    input="$inp_txt $inp_txt $inp_txt"
                fi
                $runner "paste $input | $cmd" -o $out
            }

            # Original stock implementation
            out="$pref.$m.orig.${n_gpus}gpus.stats.json"
            [[ -s $out ]] || {
                if [[ $m == "bleurt-20" ]]; then
                    # original bleurt has no multi gpu support
                    [[ $n_gpus -gt 1 ]] && continue
                    cmd="python -m bleurt.score_files --bleurt_batch_size $BATCH_SIZE --batch_same_length=True --candidate_file $inp_txt --reference_file $inp_txt --bleurt_checkpoint $ROOT/BLEURT-20"
                else
                    cmd="comet-score --model Unbabel/$m $data_args --gpus $n_gpus --batch_size $BATCH_SIZE"
                fi
                $runner "$cmd" -o $out
            }
        done
    done
}


benchmark(){
    local cmd="$1"
    local out="$2"
    local n_runs=$N_RUNS
    [[ $out =~ stats.json$ ]] || out="$out.stats.json"
    if [[ -s $out ]]; then
        log "Skipping $cmd; stats exists at $out"
    else
        log "Benchmarking $cmd; stats=$out"
        hyperfine --runs $n_runs --export-json $out "$cmd"
    fi
}

run_bechmarks(){
    local data=$1
    log "Data is ready at $DATA. Total lines: $(wc -l < $data)"
    [[ -s $data ]] || {
        log "Error: data file $data is empty"
        exit 1
    }
    cut -f1 $data | sort | uniq -c >&2
    src_txt="${data%.tsv}.src.txt"
    ref_txt="${data%.tsv}.ref.txt"
    mt_txt="${data%.tsv}.mt.txt"
    data_args="-s $src_txt -r $ref_txt -t $mt_txt"

    while IFS=":" read col path; do
       [[ -s $path ]] || {
            log "Extracting $path"
            rm -f $path $path.tmp
            cut -f$col $data > $path.tmp && mv $path{.tmp,}
       }
    done < <(echo -e "3:$src_txt\n4:$ref_txt\n5:$mt_txt")\

    ls -alh $src_txt $ref_txt $mt_txt >&2

    for n_gpus in 8 1; do
        gpu_ids="$(seq 0 $((n_gpus-1)) | tr '\n' ' ')"
        export CUDA_VISIBLE_DEVICES="$(seq 0 $((n_gpus-1)) | tr '\n' ',' | sed 's/,$//')"

        for m in ${METRICS[@]}; do
            # Pymarian
            out="${data%.tsv}.scores.$m.pymarian.${n_gpus}gpus"
            cmd="pymarian-eval --devices $gpu_ids -m $m $data_args --mini-batch $BATCH_SIZE"
            benchmark "$cmd -o $out.\$(date +%y%m%d%H%M%S).txt" "$out"

            out="${data%.tsv}.scores.$m.pymarian_fp16.${n_gpus}gpus"
            benchmark "$cmd --fp16 -o $out.\$(date +%y%m%d%H%M%S).txt" "$out"

            # the c++ version; get cmd from pymarian --print-cmd
            cmd=$(pymarian-eval --print-cmd --devices $gpu_ids -m $m $data_args --mini-batch $BATCH_SIZE)
            out="${data%.tsv}.scores.$m.marianbin.${n_gpus}gpus"
            #we need to decide input fields based on the model --like
            if [[ "$cmd " =~ "--like bleurt " ]]; then
                input="$mt_txt $ref_txt"
            elif [[ "$cmd " =~ "--like comet " ]]; then
                input="$src_txt $mt_txt $ref_txt"
            elif [[ "$cmd " =~ "--like comet-qe " ]]; then
                input="$src_txt $mt_txt"
            else
                log "Unknown model type: $cmd"
                exit 1
            fi
            #hyperfine --runs $N_RUNS --export-json $out.stats.json "paste $input | $cmd > $out.\$(date +%y%m%d%H%M%S).txt"
            benchmark "paste $input | $cmd > $out.\$(date +%y%m%d%H%M%S).txt" "$out"

            # Original stock implementation
            out="${data%.tsv}.scores.$m.orig.${n_gpus}gpus"
            # original bleurt has no multi gpu support
            [[ $m == "bleurt-20" && $n_gpus -gt 1 ]] && continue
            if [[ $m == "bleurt-20" ]]; then
                cmd="python -m bleurt.score_files --bleurt_batch_size $BATCH_SIZE --batch_same_length=True --candidate_file $mt_txt --reference_file $ref_txt --bleurt_checkpoint $ROOT/BLEURT-20 --scores_file $out.\$(date +%y%m%d%H%M%S).txt"
            else
                # adjust batch size for large models. comet-score raises cuda OOM error for large batch sizes
                bsize=$BATCH_SIZE
                [[ $m == "wmt23-cometkiwi-da-xl" && $bsize -gt 64 ]] && bsize=64
                cmd="comet-score --model Unbabel/$m $data_args --gpus $n_gpus --batch_size $bsize --to_json $out.\$(date +%y%m%d%H%M%S).json"
            fi
            # [[ $RUN_WARMUP -eq 1 ]] && warmup orig $m
            benchmark "$cmd" "$out"
        done
    done
}

main(){
    set -u
    check_tools
    get_data
    #measure_loading
    #results_loading
    #exit 0
    if [[ -z $MAX_SEGS ]]; then # use full data
        data=$DATA
    else # use a subset of data
        data=${DATA/all/subset}
        data=${data%.tsv}.$MAX_SEGS.tsv
        [[ -s $data ]] || {
            shuf -n $MAX_SEGS $DATA > $data.tmp && mv $data{.tmp,}
        }
    fi
    run_bechmarks $data
}
main "$@"