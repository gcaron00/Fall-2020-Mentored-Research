from function import show_slice, show_activations, plot_roc
import pickle
import numpy as np
import os

files = list(os.scandir('./run_opt_cv_opts'))
files[0].is_file():
f = open(files[i], "rb")
opts = pickle,load(f) 
f.close()        
i = 0
if (opts['T']['ytest'][0][i] == 0):
    subject_type = 'patient'
else:
    subject_type = 'control'
plt1 = show_slice(opts['T']['xtest'][i, :, :, 0], subject_type, opts['jj'])
plt1.savefig('./images/opt_slice.png')

os.mkdir('./Images/opt_cv_activations')
f = open('./opt_cv_results/opt_cv_A.pkl', "rb")
A = pickle.load(f) 
f.close()
fig1, fig2, fig3 = show_activation(A)
fig1.savefig('./images/opt_cv_activations/conv_1_activations.png')
fig2.savefig('./images/opt_cv_activations/conv_2_activations.png')
fig3.savefig('./images/opt_cv_activations/conv_3_activations.png')

f = open('./opt_cv_results/opt_cv_C.pkl', "rb")
C = pickle.load(f) 
f.close()
plt3 = plot_roc(C)
plt3.savefig('./images/roc_curve_plot.png')

os.mkdir('./images/slice_activations')
f = open('./slice_results/slice_C.pkl', "rb")
C_slice = pickle.load(f) 
f.close()
for i in C:
    os.mkdir('./images/slice_activations/slice_{0:d}'.format(i['slice']))
    fig1, fig2, fig3 = show_activations(i['A'])
    fig1.savefig('./images/slice_activations/slice_{0:d}/slice_{1:d}_conv_1_activations'.format(i['slice'], i['slice']))
    fig2.savefig('./images/slice_activations/slice_{0:d}/slice_{1:d}_conv_2_activations'.format(i['slice'], i['slice']))
    fig3.savefig('./images/slice_activations/slice_{0:d}/slice_{1:d}_conv_3_activations'.format(i['slice'], i['slice']))