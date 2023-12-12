import numpy as np
import matplotlib.pyplot as plt

# Read the CSV files
original = np.genfromtxt('original.csv', delimiter=',')
rms = np.genfromtxt('rms.csv', delimiter=',')
mix = np.genfromtxt('mix.csv', delimiter=',')

# # Plot the data
plt.plot(original[:,0], label='Original train',color='blue')
plt.plot(original[:,1], linestyle='--', label='Original test',color='blue')
plt.plot(rms[:,0], label='RMS train',color='red')
plt.plot(rms[:,1], linestyle='--', label='RMS test',color='red')
plt.plot(mix[:,0], label='Mix train',color='green')
plt.plot(mix[:,1], linestyle='--', label='Mix test',color='green')

# Label the axes
plt.xlabel('time')
plt.ylabel('error')

# Add a legend
plt.legend()

# Save the figure
plt.savefig('comparison.png')

# Show the plot
plt.show()

# Label the axes
plt.xlabel('time')
plt.ylabel('error')

# Add a legend
plt.legend()

# Save the figure
plt.savefig('comparison.png')

# Show the plot
plt.show()


