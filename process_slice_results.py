from functions import process_slice
import pickle
import numpy as np

opts_dir = './run_slice_opts'
nets_dir = './run_slice_nets'
C, pslice, metric_stats = process_slice(opts_dir, nets_dir)

f = open('./slice_results/slice_C.pkl',"wb")
pickle.dump(C, f)
f.close()

f = open('./slice_results/slice_C.txt',"w")
f.write(str(C))
f.close()

f = open('./slice_results/slice_pslice.pkl',"wb")
pickle.dump(pslice, f)
f.close()

f = open('./slice_results/slice_pslice.txt',"w")
f.write(str(pslice))
f.close()

f = open('./slice_results/slice_metric_stats.pkl',"wb")
pickle.dump(metric_stats, f)
f.close()

f = open('./slice_results/slice_metric_stats.txt',"w")
f.write(str(metric_stats))
f.close()

