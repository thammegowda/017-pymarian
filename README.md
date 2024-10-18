# Pymarian Benchmark Scripts

This repository contains scripts to reproduce results presented in our PyMarian paper @ EMNLP2024 Demos.


* Peer-review: https://openreview.net/forum?id=3BKsyqIieh
* ArXiv: https://arxiv.org/abs/2408.11853
* Aclweb : TODO: 


## Files
* `benchmarks.sh`  -- the main benchmark script that downloads data, runs pymarian, comet-score or bleurt and creates score files
* `memorybench.py`  -- used by benchmarks.sh to benchmark loading time and memory utilization
* `requirements.txt`  -- 
* `report.py`  -- for generating time and speedup tables from the output of benchmarks.sh. Use `-f latex` to get latex tables
* `warmup-results.sh` -- for generating warmup results  table
* `pymarian.ipynb` -- notebook for pymarian API



The outputs from our benchmark run are available in [releases/download/emnlp24-demo/wmt23.tgz](https://github.com/thammegowda/017-pymarian/releases/download/emnlp24-demo/wmt23.tgz)
