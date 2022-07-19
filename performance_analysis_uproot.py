import uproot
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import multiprocessing
import math
import csv

path = "../merged/"

def get_total_size(path, n_files):
    filenames = sorted(os.listdir(path))
    total_size = sum([os.path.getsize(path + filenames[i]) for i in range(n_files)])
    return total_size / (2**30)

def col_average(data):
    n_rows = len(data)
    n_cols = len(data[0])
    return [sum([data[i][j] for i in range(1, n_rows)]) / (n_rows - 1) for j in range(n_cols)]

def col_standard_deviation(data):
    n_rows = len(data)
    n_cols = len(data[0])
    mean = col_average(data)
    return [(sum([(data[i][j] - mean[j])**2 for i in range(1, n_rows)]) / (n_rows - 1))**0.5 for j in range(n_cols)]

def read_part_of_data(files, n_entries, index, result):
    data = []
    for i in range(len(files)):
        data.append(files[i].arrays("candidate_vMass", 
                              "(candidate_charge == 0)\
                              & (candidate_cosAlpha > 0.99)\
                              & (candidate_lxy / candidate_lxyErr > 3.0)\
                              & (candidate_vProb > 0.05)\
                              & (ditrack_mass > 1.014) & (ditrack_mass < 1.024)\
                              & (candidate_vMass > 5.33) & (candidate_vMass < 5.4)",
                              entry_start=n_entries[i] * index,
                              entry_stop=n_entries[i] * (index + 1),
                              array_cache=None,
                              library="np")["candidate_vMass"])
    result.append(np.concatenate(tuple(data)))

def runtime_measure_mp(path, n_files, n_processes):
    start = time.time()
    files = [uproot.open(path=path + filename + ":rootuple/CandidateTree", object_cache=None, array_cache=None) for filename in sorted(os.listdir(path))[:n_files]]
    n_entries = [math.ceil(file.num_entries / n_processes) for file in files]
    result = multiprocessing.Manager().list()
    processes = []
    for i in range(n_processes):
        p = multiprocessing.Process(target=read_part_of_data, args=[files, n_entries, i, result])
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    np.concatenate(tuple(result))
    
    return time.time() - start

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
    while not bool(slices) or slices[-1][-1] != (file_entries[-1]):
        slices.append([file_start, entry_start] + partition_helper(slice_entries, file_entries, file_start, entry_start))
        file_start = slices[-1][-2]
        entry_start = slices[-1][-1]
    return slices

def read_part_of_data_test(files, partitions, index, result):
    data = []
    for i in range(partitions[index][0], partitions[index][2] + 1):
        data.append(files[i].arrays("candidate_vMass", 
                              "(candidate_charge == 0)\
                              & (candidate_cosAlpha > 0.99)\
                              & (candidate_lxy / candidate_lxyErr > 3.0)\
                              & (candidate_vProb > 0.05)\
                              & (ditrack_mass > 1.014) & (ditrack_mass < 1.024)\
                              & (candidate_vMass > 5.33) & (candidate_vMass < 5.4)",
                              entry_start=partitions[index][1] if i == partitions[index][0] else None,
                              entry_stop=partitions[index][3] + 1 if i == partitions[index][2] else None,
                              array_cache=None,
                              library="np")["candidate_vMass"])
    result.append(np.concatenate(tuple(data)))

def runtime_measure_mp_test(path, n_files, n_processes):
    start = time.time()
    files = [uproot.open(path=path + filename + ":rootuple/CandidateTree", object_cache=None, array_cache=None) for filename in sorted(os.listdir(path))[:n_files]]
    partitions = partition(files, n_processes)
    result = multiprocessing.Manager().list()
    processes = []
    for i in range(n_processes):
        p = multiprocessing.Process(target=read_part_of_data_test, args=[files, partitions, i, result])
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    np.concatenate(tuple(result))
    
    return time.time() - start
    
def runtime_vs_size_mp(path, n_processes, max_files, step, n_loops, target_dir):
    result_path = ("runtime_tests_uproot/%s/runtime_vs_size_mp_%d_%d_%d_%d.csv" % (target_dir, n_processes, max_files, step, n_loops))
    x = [get_total_size(path, a) for a in range(step, max_files + step, step)]
    with open(result_path, "w+", newline="") as f:
        csv.writer(f).writerow(x)
    for n in range(n_loops):
        y = [runtime_measure_mp(path, i, n_processes) for i in range(step, max_files + step, step)]
        with open(result_path, "a+", newline="") as f:
            csv.writer(f).writerow(y)

def runtime_vs_size_plot_mp(path, n_processes, max_files, step, n_loops, target_dir):
    if not os.path.exists("runtime_tests_uproot/%s" % (target_dir)):
        os.mkdir("runtime_tests_uproot/%s" % (target_dir))
    if not os.path.exists("figures/uproot/%s" % (target_dir)):
        os.mkdir("figures/uproot/%s" % (target_dir))
    result_path = ("runtime_tests_uproot/%s/runtime_vs_size_mp_%d_%d_%d_%d.csv" % (target_dir, n_processes, max_files, step, n_loops))
    if not os.path.exists(result_path):
        runtime_vs_size_mp(path, n_processes, max_files, step, n_loops, target_dir)
    with open(result_path, "r") as f:
        data = [[float(a) for a in row] for row in csv.reader(f)]
        plt.figure(figsize = (20, 8))
        plt.title('Runtime vs Size (%d processes)' % (n_processes))
        plt.xlabel('Size (GB)')
        plt.ylabel('Runtime (s)')
        #plt.scatter(data[0], col_average(data))
        #for r in range(1, len(data)):
            #plt.scatter(data[0], data[r])
        plt.errorbar(data[0], col_average(data[1:]), yerr=col_standard_deviation(data[1:]), fmt="o", ecolor="orange")
        for i in range(len(data[0])): 
            plt.annotate("%.2f(%d)" % (round(col_average(data[1:])[i], 2), (i + 1) * step), (data[0][i], col_average(data[1:])[i]))
        plt.savefig('figures/uproot/%s/runtime_vs_size_mp_%d_%d_%d_%d.png' % (target_dir, n_processes, max_files, step, n_loops), bbox_inches='tight')
        
def runtime_vs_processes(path, n_files, max_processes, step, n_loops, target_dir):
    parent_dir = path.split('/')[-2]
    result_path = ("runtime_tests_uproot/%s/runtime_vs_processes_%d_%d_%d_%d.csv" % (target_dir, n_files, max_processes, step, n_loops))
    x = [a for a in range(step, max_processes + step, step)]
    with open(result_path, "w+", newline="") as f:
        csv.writer(f).writerow(x)
    for n in range(n_loops):
        y = [runtime_measure_mp(path, n_files, i) for i in range(step, max_processes + step, step)]
        with open(result_path, "a+", newline="") as f:
            csv.writer(f).writerow(y)
            
def runtime_vs_processes_plot(path, n_files, max_processes, step, n_loops, target_dir):
    parent_dir = path.split('/')[-2]
    if not os.path.exists("runtime_tests_uproot/%s" % (target_dir)):
        os.mkdir("runtime_tests_uproot/%s" % (target_dir))
    if not os.path.exists("figures/uproot/%s" % (target_dir)):
        os.mkdir("figures/uproot/%s" % (target_dir))
    result_path = ("runtime_tests_uproot/%s/runtime_vs_processes_%d_%d_%d_%d.csv" % (target_dir, n_files, max_processes, step, n_loops))
    if not os.path.exists(result_path):
        runtime_vs_processes(path, n_files, max_processes, step, n_loops, target_dir)
    with open(result_path, "r") as f:
        data = [[float(a) for a in row] for row in csv.reader(f)]
        plt.figure(figsize = (20, 8))
        plt.title('Runtime vs Processes (%d files, %.2f GB)' % (n_files, get_total_size(path, n_files)))
        plt.xlabel('Processes')
        plt.ylabel('Runtime (s)')
        #plt.scatter(data[0], col_average(data))
        #for r in range(1, len(data)):
            #plt.scatter(data[0], data[r])
        plt.xticks(range(step, max_processes + 1, step))
        plt.errorbar(data[0], col_average(data[:]), yerr=col_standard_deviation(data[:]), fmt="o", ecolor="orange")
        for i in range(len(data[0])): 
            plt.annotate(round(col_average(data[:])[i], 2), (data[0][i], col_average(data[:])[i]))
        plt.savefig('figures/uproot/%s/runtime_vs_processes_%d_%d_%d_%d.png' % (target_dir, n_files, max_processes, step, n_loops), bbox_inches='tight')
        
def runtime_vs_processes_test(path, n_files, max_processes, step, n_loops, target_dir):
    parent_dir = path.split('/')[-2]
    result_path = ("runtime_tests_uproot/%s/runtime_vs_processes_%d_%d_%d_%d.csv" % (target_dir, n_files, max_processes, step, n_loops))
    x = [a for a in range(step, max_processes + step, step)]
    with open(result_path, "w+", newline="") as f:
        csv.writer(f).writerow(x)
    for n in range(n_loops):
        y = [runtime_measure_mp_test(path, n_files, i) for i in range(step, max_processes + step, step)]
        with open(result_path, "a+", newline="") as f:
            csv.writer(f).writerow(y)
            
def runtime_vs_processes_plot_test(path, n_files, max_processes, step, n_loops, target_dir):
    parent_dir = path.split('/')[-2]
    if not os.path.exists("runtime_tests_uproot/%s" % (target_dir)):
        os.mkdir("runtime_tests_uproot/%s" % (target_dir))
    if not os.path.exists("figures/uproot/%s" % (target_dir)):
        os.mkdir("figures/uproot/%s" % (target_dir))
    result_path = ("runtime_tests_uproot/%s/runtime_vs_processes_%d_%d_%d_%d.csv" % (target_dir, n_files, max_processes, step, n_loops))
    if not os.path.exists(result_path):
        runtime_vs_processes_test(path, n_files, max_processes, step, n_loops, target_dir)
    with open(result_path, "r") as f:
        data = [[float(a) for a in row] for row in csv.reader(f)]
        plt.figure(figsize = (20, 8))
        plt.title('Runtime vs Processes (%d files, %.2f GB)' % (n_files, get_total_size(path, n_files)))
        plt.xlabel('Processes')
        plt.ylabel('Runtime (s)')
        #plt.scatter(data[0], col_average(data))
        #for r in range(1, len(data)):
            #plt.scatter(data[0], data[r])
        plt.xticks(range(step, max_processes + 1, step))
        plt.errorbar(data[0], col_average(data[:]), yerr=col_standard_deviation(data[:]), fmt="o", ecolor="orange")
        for i in range(len(data[0])): 
            plt.annotate(round(col_average(data[:])[i], 2), (data[0][i], col_average(data[:])[i]))
        plt.savefig('figures/uproot/%s/runtime_vs_processes_%d_%d_%d_%d.png' % (target_dir, n_files, max_processes, step, n_loops), bbox_inches='tight')
        
# runtime_vs_processes_plot(path, 60, 60, 4, 10, "merged")
# runtime_vs_processes_plot(path, 80, 80, 5, 10, "merged")

runtime_vs_processes_plot_test(path, 60, 60, 4, 10, "merged_test")
