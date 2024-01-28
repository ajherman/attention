import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dir',type=str,default='.', help='Directory of CSV files')
parser.add_argument('--files', nargs='+', help='List of CSV files')
parser.add_argument('--save-name',type=str, default='compare_results',help='Where to save the plots')
colors=['blue','red','green','orange','purple','brown','pink','gray','olive','cyan']
args = parser.parse_args()

if args.files is None:     # Find all CSV files in the current directory
    args.files = []
    for file_name in os.listdir(args.dir):
        if file_name.endswith('.csv'):
            args.files.append(file_name[:-4])

for idx,f_name in enumerate(args.files):
    if f_name.endswith('.csv'):
        f_name = f_name[:-4]
    file_path = args.dir+'/'+f_name+'.csv'
    if not os.path.exists(file_path):
        print(f"File '{file_path}' does not exist. Skipping...")
    else:
        try:
            data = np.genfromtxt(file_path, delimiter=',')
            plt.plot(data[:,0]+0.005*np.random.normal(), label=f_name+' train', linestyle='--', color=colors[idx], linewidth=0.5)
            plt.plot(data[:,1]+0.005*np.random.normal(), label=f_name+' test',color=colors[idx], linewidth=0.5)
        except:
            print(f"Could not plot '{file_path}'")

# Label the axes
plt.xlabel('time')
plt.ylabel('error')
plt.title('Error vs Time')

# Add a legend
plt.legend(fontsize='small')

# Save the figure
plt.savefig(args.dir+'/'+args.save_name+'.png')

# Show the plot
plt.show()



