import numpy as np
import matplotlib.pyplot as plt

# Read the CSV files
original = np.genfromtxt('original.csv', delimiter=',')
rms = np.genfromtxt('rms.csv', delimiter=',')
mix = np.genfromtxt('mix.csv', delimiter=',')

# # Plot the data
plt.plot(original[:,0])
# plt.plot(original[:,1])
plt.plot(rms[:,0])
plt.plot(mix[:,0])


# plt.plot(df2)
# print(df1)
# Label the axes
plt.xlabel('time')
plt.ylabel('error')

# Add a legend
plt.legend()

# Save the figure
plt.savefig('comparison.png')

# Show the plot
plt.show()


