import numpy as np
import matplotlib.pyplot as plt

# Read the CSV files
df1 = np.genfromtxt('original.csv', delimiter=',')
# df2 = np.genfromtxt('alternate.csv', delimiter=',')

# # Plot the data
plt.plot(df1)
# plt.plot(df2)
print(df1)
# Label the axes
plt.xlabel('time')
plt.ylabel('error')

# Add a legend
plt.legend()

# Save the figure
plt.savefig('/home/ari/learning/comparison.png')

# Show the plot
plt.show()


