import csv
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import ROOT
import time

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
 
def partition(path, n_files, n_processes):
    filenames = sorted(os.listdir(path))
    partitions = []
    curr = 0
    for i in range(n_processes):
        if i >= n_files: break
        n_files_in_partition = n_files // n_processes if i >= n_files % n_processes else n_files // n_processes + 1
        files_to_read = ROOT.std.vector('string')()
        for j in range(n_files_in_partition):
            files_to_read.push_back(path + filenames[curr + j])
        curr += n_files_in_partition
        partitions.append(files_to_read)
    return partitions

def to_numpy(files, result):
    if files.empty(): return
#     ROOT.ROOT.DisableImplicitMT()
    data = ROOT.RDataFrame("rootuple/CandidateTree", files)
    cut = data.Filter("candidate_charge == 0")\
          .Filter("candidate_cosAlpha > 0.99")\
          .Filter("candidate_vProb > 0.05")\
          .Filter("candidate_lxy / candidate_lxyErr > 3.0")\
          .Filter("ditrack_mass > 1.014")\
          .Filter("ditrack_mass < 1.024")\
          .Filter("candidate_vMass > 5.33")\
          .Filter("candidate_vMass < 5.40")
    result.append(cut.AsNumpy(["candidate_vMass"])["candidate_vMass"])

def runtime_measure(path, n_files, mt):
    if n_files == 0: return 0
    # Specify the number of threads
    if mt: ROOT.ROOT.EnableImplicitMT()
    
    # Get paths to all the files to be read 
    filenames = sorted(os.listdir(path))
    files_to_read = ROOT.std.vector('string')()
    for i in range(n_files):
        files_to_read.push_back(path + filenames[i])
    
    # Measure runtime
    start_time = time.time()
    
    data = ROOT.RDataFrame("rootuple/CandidateTree", files_to_read)
    cut = data.Filter("candidate_charge == 0")\
          .Filter("candidate_cosAlpha > 0.99")\
          .Filter("candidate_vProb > 0.05")\
          .Filter("candidate_lxy / candidate_lxyErr > 3.0")\
          .Filter("ditrack_mass > 1.014")\
          .Filter("ditrack_mass < 1.024")\
          .Filter("candidate_vMass > 5.33")\
          .Filter("candidate_vMass < 5.40")
    np_array = cut.AsNumpy(["candidate_vMass"])
    
    return time.time() - start_time

def runtime_measure_mp(path, n_files, n_processes):
    if n_files == 0: return 0
    if n_processes == 0: return runtime_measure(path, n_files, False)
    start_time = time.time()
    partitions = partition(path, n_files, n_processes)
    processes = []
    result = multiprocessing.Manager().list()
    for files in partitions:
        p = multiprocessing.Process(target=to_numpy, args=[files, result])
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
     
    np.concatenate(tuple(result))
    runtime = time.time() - start_time
    
    return runtime

def runtime_vs_variable(path, target_dir, measure_function, variable, step, n_loops, var_max, constant=None):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    result_path = ("%s/runtime_vs_%s_%d_%d_%d_%d.csv" % (target_dir, variable, constant, var_max, step, n_loops)) if constant else ("%s/runtime_vs_%s_%d_%d_%d.csv" % (target_dir, variable, var_max, step, n_loops))
    
    x = [get_total_size(path, a) for a in range(0, var_max + step, step)] if "size" in variable else [a for a in range(0, var_max + step, step)]
        
    with open(result_path, "w+", newline="") as f:
        csv.writer(f).writerow(x)
    for n in range(n_loops):
        y = [measure_function(*(path, i if variable == "size" else constant, constant if variable == "size" else i) if constant else (path, i, True)) for i in range(0, var_max + step, step)]
        with open(result_path, "a+", newline="") as f:
            csv.writer(f).writerow(y)
            

path = "/home/hdhoang2001/data/128_files/"
target_dir = "runtime_tests_rdf/test"

runtime_vs_variable(path, target_dir, runtime_measure_mp, "processes", 4, 10, 128, 128)
runtime_vs_variable(path, target_dir, runtime_measure_mp, "size_mp", 4, 10, 128, 64)
runtime_vs_variable(path, target_dir, runtime_measure_mp, "size_mp", 4, 10, 128, 32)
runtime_vs_variable(path, target_dir, runtime_measure, "size", 4, 10, 128)
# runtime_vs_processes_plot(path, 60, 5, 10, "64_files")
# runtime_vs_processes_plot(path, 128, 4, 11, "128_files")
# runtime_vs_processes_plot(path, 96, 4, 11, "128_files")
# runtime_vs_processes_plot(path, 64, 4, 11, "128_files")
# runtime_vs_size_plot_mp(path, 64, 128, 4, 11, "128_files")
# runtime_vs_size_plot_mp(path, 32, 128, 4, 11, "128_files")
# runtime_vs_size_plot_mp(path, 16, 128, 4, 11, "128_files")
# runtime_vs_size_plot(path, 128, 4, 11, "128_files")



