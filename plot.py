import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dir',type=str,default='.', help='Directory of CSV files')
parser.add_argument('--files', nargs='+', help='List of CSV files')
parser.add_argument('--save-name',type=str, help='Where to save the plots')
# parser.add_argument('--save-name',type=str, help='Name of the plot')
colors=['blue','red','green','orange','purple','brown','pink','gray','olive','cyan']
args = parser.parse_args()

for idx,f_name in enumerate(args.files):
    if f_name.endswith('.csv'):
        f_name = f_name[:-4]
    file_path = args.dir+'/'+f_name+'.csv'
    if not os.path.exists(file_path):
        print(f"File '{file_path}' does not exist. Skipping...")
    else:
        try:
            data = np.genfromtxt(file_path, delimiter=',')
            plt.plot(data[:,0], label=f_name+' train',linestyle='--',color=colors[idx])
            plt.plot(data[:,1], label=f_name+' test',color=colors[idx])
        except:
            print(f"Could not plot '{file_path}'")


# # Read the CSV files
# original = np.genfromtxt('original.csv', delimiter=',')
# rms = np.genfromtxt('rms.csv', delimiter=',')
# mix = np.genfromtxt('mix.csv', delimiter=',')

# # # Plot the data
# plt.plot(original[:,0], label='Original train',color='blue')
# plt.plot(original[:,1], linestyle='--', label='Original test',color='blue')
# plt.plot(rms[:,0], label='RMS train',color='red')
# plt.plot(rms[:,1], linestyle='--', label='RMS test',color='red')
# plt.plot(mix[:,0], label='Mix train',color='green')
# plt.plot(mix[:,1], linestyle='--', label='Mix test',color='green')

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



