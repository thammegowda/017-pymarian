# Pymarian Benchmark Scripts

* benchmarks.sh  -- the main benchmark script that downloads data, runs pymarian, comet-score or bleurt and creates score files
* memorybench.py  -- used by benchmarks.sh to benchmark loading time and memory utilization
* requirements.txt  -- 
* report.py  -- for generating time and speedup tables from the output of benchmarks.sh. Use `-f latex` to get latex tables
* warmup-results.sh -- for generating warmup results  table
* pymarian.ipynb -- notebook for pymarian API
