import csv
import math
import multiprocessing
import numpy as np
import os
import sys
import time
import uproot

def partition_helper(slice_entries, file_entries, file_curr, entry_curr):
    if slice_entries <= file_entries[file_curr] - entry_curr:
        return [file_curr, slice_entries + entry_curr]
    elif file_curr == len(file_entries) - 1:
        return [file_curr, file_entries[-1]]
    else:
        return partition_helper(slice_entries - file_entries[file_curr] + entry_curr, file_entries, file_curr + 1, 0)

def partition(file_entries, n_processes):
    slice_entries = math.ceil(sum(file_entries) / n_processes)
    slices = []
    file_start = 0
    entry_start = 0
    for i in range(n_processes):
        slices.append([file_start, entry_start] + partition_helper(slice_entries, file_entries, file_start, entry_start))
        file_start = slices[-1][-2]
        entry_start = slices[-1][-1]
    return slices

def read_slice(paths, slices, index, result):
    data_slice = []
    for i in range(slices[index][0], slices[index][2] + 1):
        with uproot.open(paths[i], object_cache=None, array_cache=None) as file:
            data_slice.append(file.arrays("candidate_vMass", 
                                  "(candidate_charge == 0)\
                                  & (candidate_cosAlpha > 0.99)\
                                  & (candidate_lxy / candidate_lxyErr > 3.0)\
                                  & (candidate_vProb > 0.05)\
                                  & (ditrack_mass > 1.014) & (ditrack_mass < 1.024)\
                                  & (candidate_vMass > 5.33) & (candidate_vMass < 5.4)",
                                  entry_start=slices[index][1] if i == slices[index][0] else None,
                                  entry_stop=slices[index][3] if i == slices[index][2] else None,
                                  array_cache=None,
                                  library="np")["candidate_vMass"])
    result.append(np.concatenate(tuple(data_slice)))
    
def runtime_measure_mp(path, n_files, n_processes):
    if n_files == 0: return 0
    if n_processes == 0: return runtime_measure(path, n_files)
    start = time.time()
    paths = [path + filename + ":rootuple/CandidateTree" for filename in sorted(os.listdir(path))[:n_files]]
    file_entries = [n[2] for n in uproot.num_entries([p for p in paths])]
    slices = partition(file_entries, n_processes)
    result = multiprocessing.Manager().list()
    processes = []
    for i in range(n_processes):
        p = multiprocessing.Process(target=read_slice, args=[paths, slices, i, result])
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    np.concatenate(tuple(result))
    
    return time.time() - start

def runtime_measure(path, n_files):
    if n_files == 0: return 0
    start = time.time()
    paths = [path + filename + ":rootuple/CandidateTree" for filename in sorted(os.listdir(path))[:n_files]]
    data = []
    for p in paths:
        with uproot.open(p, object_cache=None, array_cache=None) as file:
            data.append(file.arrays("candidate_vMass", 
                                  "(candidate_charge == 0)\
                                  & (candidate_cosAlpha > 0.99)\
                                  & (candidate_lxy / candidate_lxyErr > 3.0)\
                                  & (candidate_vProb > 0.05)\
                                  & (ditrack_mass > 1.014) & (ditrack_mass < 1.024)\
                                  & (candidate_vMass > 5.33) & (candidate_vMass < 5.4)",
                                  array_cache=None,
                                  library="np")["candidate_vMass"])
        
    np.concatenate(tuple(data))
    
    return time.time() - start

def runtime_vs_variable(path, target_dir, measure_function, variable, step, n_loops, var_max, constant=None):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    result_path = ("%s/runtime_vs_%s_%d_%d_%d_%d.csv" % (target_dir, variable, constant, var_max, step, n_loops)) if constant else ("%s/runtime_vs_%s_%d_%d_%d.csv" % (target_dir, variable, var_max, step, n_loops))
    
    x = [a for a in range(0, var_max + step, step)]
    
    if not os.path.exists(result_path):
        with open(result_path, "w", newline="") as f:
            csv.writer(f).writerow(x)
    for n in range(n_loops):
        with open(result_path, "r") as f:
            if sum(1 for row in csv.reader(f)) == n_loops + 1: break
        y = [measure_function(*(path, i if "size" in variable else constant, constant if "size" in variable else i) if constant else (path, i)) for i in range(0, var_max + step, step)]
        with open(result_path, "a", newline="") as f:
            csv.writer(f).writerow(y)
        
path = "../data/32_files/"
target_dir = "runtime_tests_uproot/32_files/" + str(sys.argv[1])

runtime_vs_variable(path, target_dir, runtime_measure_mp, "processes", 4, 20, 128, 32)
runtime_vs_variable(path, target_dir, runtime_measure_mp, "size_mp", 1, 20, 32, 64)
runtime_vs_variable(path, target_dir, runtime_measure_mp, "size_mp", 1, 20, 32, 32)
runtime_vs_variable(path, target_dir, runtime_measure, "size", 1, 20, 32)



