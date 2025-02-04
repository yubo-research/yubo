#!/usr/bin/env python

if __name__ == "__main__":
    from problems.benchmark_functions import all_benchmarks

    for fn in all_benchmarks():
        print(fn)
