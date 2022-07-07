import ROOT
import os
import time
import matplotlib.pyplot as plt
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

def runtime_measure(path, n_files, n_threads):
    # Specify the number of threads
    ROOT.ROOT.EnableImplicitMT(n_threads)
    
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
    runtime = time.time() - start_time
    
    return runtime
    
def runtime_vs_size(path, n_threads, max_files, step, n_loops, target_dir):
    result_path = ("runtime_tests_rdf/%s/runtime_vs_size_%d_%d_%d_%d.csv" % (target_dir, n_threads, max_files, step, n_loops))
    x = [get_total_size(path, a) for a in range(step, max_files + step, step)]
    with open(result_path, "w+", newline="") as f:
        csv.writer(f).writerow(x)
    for n in range(n_loops):
        y = [runtime_measure(path, i, n_threads) for i in range(step, max_files + step, step)]
        with open(result_path, "a+", newline="") as f:
            csv.writer(f).writerow(y)

def runtime_vs_size_plot(path, n_threads, max_files, step, n_loops, target_dir):
    if not os.path.exists("runtime_tests_rdf/%s" % (target_dir)):
        os.mkdir("runtime_tests_rdf/%s" % (target_dir))
    if not os.path.exists("figures/%s" % (target_dir)):
        os.mkdir("figures/%s" % (target_dir))
    result_path = ("runtime_tests_rdf/%s/runtime_vs_size_%d_%d_%d_%d.csv" % (target_dir, n_threads, max_files, step, n_loops))
    if not os.path.exists(result_path):
        runtime_vs_size(path, n_threads, max_files, step, n_loops, target_dir)
    with open(result_path, "r") as f:
        data = [[float(a) for a in row] for row in csv.reader(f)]
        plt.figure(figsize = (15, 5))
        plt.title('Runtime vs Size (%d threads)' % (n_threads))
        plt.xlabel('Size (GB)')
        plt.ylabel('Runtime (s)')
        #plt.scatter(data[0], col_average(data))
        #for r in range(1, len(data)):
            #plt.scatter(data[0], data[r])
        plt.errorbar(data[0], col_average(data), yerr=col_standard_deviation(data), fmt="o", ecolor="orange")
        plt.savefig('figures/%s/runtime_vs_size_%d_%d_%d_%d.png' % (target_dir, n_threads, max_files, step, n_loops), bbox_inches='tight')

def runtime_vs_threads(path, n_files, max_threads, step, n_loops, target_dir):
    parent_dir = path.split('/')[-2]
    result_path = ("runtime_tests_rdf/%s/runtime_vs_threads_%d_%d_%d_%d.csv" % (target_dir, n_files, max_threads, step, n_loops))
    x = [a for a in range(step, max_threads + step, step)]
    with open(result_path, "w+", newline="") as f:
        csv.writer(f).writerow(x)
    for n in range(n_loops):
        y = [runtime_measure(path, n_files, i) for i in range(step, max_threads + step, step)]
        with open(result_path, "a+", newline="") as f:
            csv.writer(f).writerow(y)

def runtime_vs_threads_plot(path, n_files, max_threads, step, n_loops, target_dir):
    parent_dir = path.split('/')[-2]
    if not os.path.exists("runtime_tests_rdf/%s" % (target_dir)):
        os.mkdir("runtime_tests_rdf/%s" % (target_dir))
    if not os.path.exists("figures/%s" % (target_dir)):
        os.mkdir("figures/%s" % (target_dir))
    result_path = ("runtime_tests_rdf/%s/runtime_vs_threads_%d_%d_%d_%d.csv" % (target_dir, n_files, max_threads, step, n_loops))
    if not os.path.exists(result_path):
        runtime_vs_threads(path, n_files, max_threads, step, n_loops, target_dir)
    with open(result_path, "r") as f:
        data = [[float(a) for a in row] for row in csv.reader(f)]
        plt.figure(figsize = (15, 5))
        plt.title('Runtime vs Threads (%.2f GB)' % (get_total_size(path, n_files)))
        plt.xlabel('Threads')
        plt.ylabel('Runtime (s)')
        #plt.scatter(data[0], col_average(data))
        #for r in range(1, len(data)):
            #plt.scatter(data[0], data[r])
        plt.errorbar(data[0], col_average(data[1:]), yerr=col_standard_deviation(data[1:]), fmt="o", ecolor="orange")
        plt.savefig('figures/%s/runtime_vs_threads_%d_%d_%d_%d.png' % (target_dir, n_files, max_threads, step, n_loops), bbox_inches='tight')
        
runtime_vs_threads_plot(path, 1, 16, 4, 10, "merged")