import numpy as np
import sys,os,pickle
import math as m
from sklearn.metrics.pairwise import euclidean_distances

import matplotlib.pyplot as plt

plt.style.use('seaborn')

names,mse,oe=sys.argv[1:],[],{}

show_graph=True
error_type="RMSE_c"
overall_errors=["RMSE_o","MAE_o"]


mse_str="info.p"
for name in names:
    #print(f"added mse for {name}")
    with open( os.path.join(name,mse_str) ,'rb') as file:
        a=pickle.load(file)
        mse.append( np.sort( a['errors'][error_type] ) )
        basename=os.path.basename(name)
        oe[basename]=[]
        for x in overall_errors:
            oe[basename].append(a['errors'][x])

colors=["grey","lightblue","lightcoral","lightgreen","rebeccapurple","purple","darkred","red","pink"]


mult=0.4
xticks=[1,10,20,30,40,50]
xlabels=[1,10,20,30,40,50]
fs=30*mult
fs2=30*0.6*mult
lw=5*mult
lw2=3*mult

fig=plt.figure()
for i in range(len(mse)):

    y=mse[i]
    x=np.arange(len(y))

    label=os.path.basename(os.path.normpath(names[i]))
    #print(f"plotting for {label}")
    plt.step(x,y,color=colors[i],label=label)

    #plt.title(label,fontsize=1.1*fs,pad=-25,fontweight="bold",color=purple)

    '''
    for i in ["top","right"]:
        ax.spines[i].set_visible(False)
    for i in ["left","bottom"]:
        ax.spines[i].set_linewidth(lw)
    '''
plt.legend()
fig.text(0.45, 0.04,"Cluster index",ha='center',va='center',fontsize=fs/mult*0.35)
fig.text(0.025, 0.45,r'RMSE_c $(kcal/mol\,\AA)(^{2})$',ha='center',va='center',fontsize=fs/mult*0.35,rotation="vertical")
#fig.suptitle(r"Mean squared force prediction error on 50 clusters""\n"r"for default model$^{[2]}$(red) and error-flattened model(blue)",y=0.995,fontsize=1*fs,fontweight="bold",color=purple)
#plt.subplots_adjust(left=0.125,top=.8,bottom=.15,right=0.9)

name="name"
labels=f"{name:<50}"
for x in overall_errors:
    labels=f"{labels}{x:<22}"
print(labels)
for x in oe.keys():
    print(f"{x:<50}{oe[x]}")

plt.savefig("compare_graph.pdf")
if show_graph:
    plt.show()









