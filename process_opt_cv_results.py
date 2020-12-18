from functions import process_results
import pickle
import numpy as np

opts_dir = './run_opt_cv_opts'
nets_dir = './run_opt_cv_nets'
C, A, metric_stats = process_results(opts_dir, nets_dir)

f = open('./opt_cv_results/opt_cv_C.pkl',"wb")
pickle.dump(C, f)
f.close() 

f = open('./opt_cv_results/opt_cv_C.txt',"w")
f.write(str(C))
f.close()

f = open('./opt_cv_results/opt_cv_A.pkl',"wb")
pickle.dump(A, f)
f.close()

f = open('./opt_cv_results/opt_cv_A.txt',"w")
f.write(str(A))
f.close()

f = open('./opt_cv_results/opt_cv_metric_stats.pkl',"wb")
pickle.dump(metric_stats, f)
f.close()

f = open('./opt_cv_results/opt_cv_metric_stats.txt',"w")
f.write(str(metric_stats))
f.close()