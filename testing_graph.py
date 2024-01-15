
import matplotlib.pyplot as plt


# Plot the main lines
plt.scatter(1.2, 2.3,)
plt.scatter(2.3, 1.4, label="Other Company")
plt.text(1.2, 2.3, "L'Oreal", fontsize=14, ha='left')
plt.text(2.3, 1.4, "Other Company", fontsize=14, ha='right')

# Add the shaded area

# Setting x-ticks, labels and plot properties
plt.xlabel("Something", fontsize=14)
plt.ylabel("Something Else", fontsize=14)

# Display the plot
plt.show()
