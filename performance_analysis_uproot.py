import csv
import math
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import sys
import time
import uproot

def get_total_size(path, n_files):
    filenames = sorted(os.listdir(path))
    total_size = sum([os.path.getsize(path + filenames[i]) for i in range(n_files)])
    return total_size / (2**30)

def get_num_entries(path, n_files):
    filenames = sorted(os.listdir(path))
    num_entries = sum([uproot.open(path + filenames[i] + ":rootuple/CandidateTree").num_entries for i in range(n_files)])
    return num_entries

def col_average(data):
    n_rows = len(data)
    n_cols = len(data[0])
    return [sum([data[i][j] for i in range(1, n_rows)]) / (n_rows - 1) for j in range(n_cols)]

def col_standard_deviation(data):
    n_rows = len(data)
    n_cols = len(data[0])
    mean = col_average(data)
    return [(sum([(data[i][j] - mean[j])**2 for i in range(1, n_rows)]) / (n_rows - 1))**0.5 for j in range(n_cols)]

def partition_helper(slice_entries, file_entries, file_curr, entry_curr):
    if slice_entries <= file_entries[file_curr] - entry_curr:
        return [file_curr, slice_entries + entry_curr]
    elif file_curr == len(file_entries) - 1:
        return [file_curr, file_entries[-1]]
    else:
        return partition_helper(slice_entries - file_entries[file_curr] + entry_curr, file_entries, file_curr + 1, 0)

def partition(files, n_processes):
    file_entries = [file.num_entries for file in files]
    slice_entries = math.ceil(sum(file_entries) / n_processes)
    slices = []
    file_start = 0
    entry_start = 0
    for i in range(n_processes):
        slices.append([file_start, entry_start] + partition_helper(slice_entries, file_entries, file_start, entry_start))
        file_start = slices[-1][-2]
        entry_start = slices[-1][-1]
    return slices

def read_slice(files, slices, index, data):
    data_slice = []
    for i in range(slices[index][0], slices[index][2] + 1):
        data_slice.append(files[i].arrays("candidate_vMass", 
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
    data.append(np.concatenate(tuple(data_slice)))

def runtime_measure_mp(path, n_files, n_processes):
    if n_files == 0: return 0
    if n_processes == 0: return runtime_measure(path, n_files)
    start = time.time()
    files = [uproot.open(path=path + filename + ":rootuple/CandidateTree", object_cache=None, array_cache=None) for filename in sorted(os.listdir(path))[:n_files]]
    slices = partition(files, n_processes)
    data = multiprocessing.Manager().list()
    processes = []
    for i in range(n_processes):
        p = multiprocessing.Process(target=read_slice, args=[files, slices, i, data])
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    np.concatenate(tuple(data))
    
    return time.time() - start

def runtime_measure(path, n_files):
    if n_files == 0: return 0
    start = time.time()
    files = [uproot.open(path=path + filename + ":rootuple/CandidateTree", object_cache=None, array_cache=None) for filename in sorted(os.listdir(path))[:n_files]]
    data = []
    for file in files:
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
        
    with open(result_path, "w+", newline="") as f:
        csv.writer(f).writerow(x)
    for n in range(n_loops):
        y = [measure_function(*(path, i if "size" in variable else constant, constant if "size" in variable else i) if constant else (path, i)) for i in range(0, var_max + step, step)]
        with open(result_path, "a+", newline="") as f:
            csv.writer(f).writerow(y)
        
path = "data/128_files/"
target_dir = "runtime_tests_uproot/" + str(sys.argv[1])

runtime_vs_variable(path, target_dir, runtime_measure_mp, "processes", 4, 20, 128, 128)
runtime_vs_variable(path, target_dir, runtime_measure_mp, "size_mp", 4, 20, 128, 64)
runtime_vs_variable(path, target_dir, runtime_measure_mp, "size_mp", 4, 20, 128, 32)
runtime_vs_variable(path, target_dir, runtime_measure, "size", 4, 20, 128)




