from functions import optimal_grid_parameters, grid_results
import pickle
import numpy as np

folder = './run_grid_opts'
D, opts, opt_params_stats = grid_results(folder)

f = open('./optimal_params/opt_params_opts.pkl',"wb")
pickle.dump(opts, f)
f.close()

f = open('./optimal_params/opt_params_opts.txt',"w")
f.write(str(opts))
f.close()

f = open('./optimal_params/opt_params_D.pkl',"wb")
pickle.dump(D, f)
f.close()

f = open('./optimal_params/opt_params_D.txt',"w")
f.write(str(D))
f.close()

f = open('./optimal_params/opt_params_stats.pkl',"wb")
pickle.dump(opt_params_stats, f)
f.close()

f = open('./optimal_params/opt_params_stats.txt',"w")
f.write(str(opt_params_stats))
f.close()