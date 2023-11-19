import matplotlib.pyplot as plt



#Interventions Graphs
"""
# Example usage:
fontsize = 14
rotation = 15

plt.scatter(2.7, 0.201)
plt.text(2.7, 0.201, "Target Network", fontsize=fontsize, ha='left', rotation=rotation)

plt.scatter(4.3, 0.214)
plt.text(4.3, 0.214, "None", fontsize=fontsize, ha='right', rotation=rotation, va='top')

plt.scatter(3.45, 0.216)
plt.text(3.45, 0.216, "EMA Network", fontsize=fontsize, ha='right', rotation=rotation, va='top')

plt.scatter(3.8, 0.232, marker="*", s=70)
plt.text(3.8, 0.232, "Trust Regions", fontsize=fontsize, ha='right', rotation=rotation, va='top')

# Adding labels and title
plt.xlabel("Policy Churn (%)")
plt.ylabel("Performance (IQM)")

# Display the plot
plt.show()
"""

"""
# Nstep vs performance - needs N=5

performance = [0.04, 0.214, 0.145, 0.103]
medians = [0.036, 0.108, 0.082, 0.067]
N = [1, 3, 10, 20]

fontsize = 14
rotation = 15
N_label = [str(i) for i in N]

plt.scatter(5, 0.218)
plt.text(5, 0.218, "10 to 3", fontsize=fontsize, ha='left', rotation=rotation, va='top')

plt.plot(N, performance, label="IQM")
plt.plot(N, medians, label="Median")
plt.xticks(N, N_label)

plt.xlabel("N-Step")
plt.ylabel("Performance")

plt.legend()

# Display the plot
plt.show()"""


"""
# gamma vs performance - needs gamma=0.95

performance = [0.232, 0.264, 0.272, 0.214]
medians = [0.183, 0.215, 0.192, 0.108]
N = [0.9, 0.95, 0.97, 0.99]


N_label = [str(i) for i in N]

plt.plot(N, performance, label="IQM")
plt.plot(N, medians, label="Median")
plt.xticks(N, N_label)

plt.xlabel("Discount Rate")
plt.ylabel("Performance")

plt.legend()

# Display the plot
plt.show()
"""

"""
# gamma vs policy churn

early = [3.6, 3.9, 4.4, 5.8]
late = [1.8, 2, 2.2, 2.8]
N = [0.9, 0.95, 0.97, 0.99]


N_label = [str(i) for i in N]

plt.plot(N, early, label="Early")
plt.plot(N, late, label="Late")
plt.xticks(N, N_label)

plt.xlabel("Discount Rate")
plt.ylabel("Policy Churn (%)")

plt.legend()

# Display the plot
plt.show()
"""

# contractive factor vs churn?

