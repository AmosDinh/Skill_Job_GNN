
# import matplotlib
# import matplotlib.patches as mpatches
# from matplotlib.patches import Rectangle

# import matplotlib.pyplot as plt
# font = {
#         'size'   : 12}

# matplotlib.rc('font', **font)
# plt.suptitle('Distribution of the absolute mean of ElbowLeftX\n per execution-file, original method',y=1.1)
# fig = plt.figure(figsize=(12,4))

# plt.subplot(211)
# plt.xlim(0,10000)
# plt.tick_params('x', labelbottom=False)
# plt.ylim(-1,1.5)
# plot_graph(df.pc1[125:]/15,skeleton_changed=df.SkeletonChanged[125:],y_true=df.label[125:],y_score=scores,is_subplot=True, scorecolor='tab:purple')
# plt.subplot(212)
# plt.xlim(0,10000)
# plt.ylim(-1,1.5)



# plot_graph(df.pc1/15,skeleton_changed=df.SkeletonChanged,y_true=df.label,y_score=pd.Series(unwindowed_scores),is_subplot=True,scorecolor='tab:orange')
# red_patch = mpatches.Patch(color='tab:blue', label='data')
# red_patch2 = mpatches.Patch(color='tab:purple', label='window scores')
# red_patch3 = mpatches.Patch(color='tab:orange', label='unwindowed scores')
# fig.legend(handles=[red_patch,red_patch2,red_patch3],loc=(0.80, 0.835),frameon=False)
# fig.suptitle('Effect of "unwindowing" scores',y=1.05)
# fig.supxlabel('time',y=-0.05)
# plt.savefig('7_2_unwindowing_data.png',dpi=300, bbox_inches = "tight")

# write a plotter like the code above, but for plotting a node degree, given a numpy array, a list or a torch tensor of node degrees (create a histogram),
# further arguments should include: title, xlabel, ylabel, savepath, savefig, showfig, patches are not needed
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy

font = {
         'size'   : 12}
matplotlib.rc('font', **font)


def plot_degree(node_degrees, title, xlabel, ylabel, savepath, savefig=False, showfig=True, xlim=None, color='blue', bins=100):
    # Convert to numpy array if it's a list or a PyTorch tensor
    if isinstance(node_degrees, list):
        node_degrees = np.array(node_degrees)
    elif isinstance(node_degrees, torch.Tensor):
        node_degrees = node_degrees.numpy()

    # calculate the bins based on the node degrees
    node_degrees = deepcopy(node_degrees)
    node_degrees = np.array(node_degrees)
    node_degrees = node_degrees[node_degrees>=xlim[0]]
    node_degrees = node_degrees[node_degrees<=xlim[1]]
        
    # Create the histogram
    plt.hist(node_degrees, bins=bins, color=color, edgecolor='black')

    # Set the title and labels
    plt.suptitle(title,y=0.98)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Set the x limits
    if xlim is not None:
        plt.xlim(xlim)

    # Set the x and y ticks to be integers
    #plt.xticks(np.arange(np.min(node_degrees), np.max(node_degrees)+1, 1.0))
    # make xticks 25 steps if xlim[1] <100, 250 steps if xlim[1] <1000, 2500 steps if xlim[1] <10000 and so on
    for i in range(1,10):
        if xlim[1] < 10**i:
            tick = 10**(i-1)
            if tick == 100:
                tick = 50
            if tick==1:
                tick=2
            x_ticks = np.arange(xlim[0], xlim[1],tick )
           
            break
    
    
    plt.xticks(x_ticks)
    #plt.yticks(np.arange(0, np.max(np.histogram(node_degrees, bins=np.arange(node_degrees.min(), node_degrees.max()+2) - 0.5)[0])+1, 1.0))

    # Save the figure if requested
    if savefig:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')

    # Show the figure if requested
    if showfig:
        plt.show()

# plot_degree(node_degrees, 'Node Degree Distribution', 'Degree', 'Count', 'degree_dist.png', savefig=True, xlim=[0, 10], color='red')
