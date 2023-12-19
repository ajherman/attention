import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dir',type=str,default='.', help='Directory of CSV files')
parser.add_argument('--files', nargs='+', help='List of CSV files')
parser.add_argument('--save-name',type=str, help='Where to save the plots')

colors=['blue','red','green','orange','purple','brown','pink','gray','olive','cyan']

args = parser.parse_args()

for idx,f_name in enumerate(args.files):
    if f_name.endswith('.csv'):
        f_name = f_name[:-4]
    file_path = args.dir+'/'+f_name+'.csv'
    if not os.path.exists(file_path):
        print(f"File '{file_path}' does not exist. Skipping...")
    else:
        data = np.genfromtxt(file_path, delimiter=',')
        plt.plot(data[:,0], label=f_name+' train',linestyle='--',color=colors[idx])
        plt.plot(data[:,1], label=f_name+' test',color=colors[idx])

# Label the axes
plt.xlabel('time')
plt.ylabel('error')
plt.title('Error vs Time')

# Add a legend
plt.legend()

# Save the figure
plt.savefig(args.dir+'/'+args.save_name+'.png')

# Show the plot
plt.show()



