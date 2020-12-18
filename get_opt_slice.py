from functions import optimal_slice, slice_results
import pickle
import numpy as np

folder = './run_slice_opts'
D, opts, opt_slice_stats = slice_results(folder)

f = open('./optimal_slice/opt_slice_opts.pkl',"wb")
pickle.dump(opts, f)
f.close() 

f = open('./optimal_slice/opt_slice_opts.txt',"w")
f.write(str(opts))
f.close()

f = open('./optimal_slice/opt_slice_D.pkl',"wb")
pickle.dump(D, f)
f.close() 

f = open('./optimal_slice/opt_slice_D.txt',"w")
f.write(str(D))
f.close()

f = open('./optimal_slice/opt_slice_stats.pkl',"wb")
pickle.dump(opt_slice_stats, f)
f.close()

f = open('./optimal_slice/opt_slice_stats.txt',"w")
f.write(str(opt_slice_stats))
f.close()