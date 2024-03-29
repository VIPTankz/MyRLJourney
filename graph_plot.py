import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import numpy as np


#Interventions Graphs
"""
# Example usage:
fontsize = 14
rotation = 15

#plt.scatter(2.7, 0.201)
plt.errorbar(2.7, 0.201, yerr=0.012, fmt='o')
plt.text(2.7, 0.195, "Target Network", fontsize=fontsize, ha='left', rotation=rotation, va='bottom')

#plt.scatter(4.3, 0.214)
plt.errorbar(4.3, 0.214, yerr=0.015, fmt='o')
plt.text(4.3, 0.214, "None", fontsize=fontsize, ha='right', rotation=rotation, va='top')

#plt.scatter(3.45, 0.216)
plt.errorbar(3.45, 0.216, yerr=0.01, fmt='o')
plt.text(3.45, 0.216, "EMA Network", fontsize=fontsize, ha='right', rotation=rotation, va='top')

#plt.scatter(3.8, 0.232, marker="*", s=70)
plt.errorbar(3.8, 0.232, yerr=0.012, fmt='o')
plt.text(3.8, 0.242, "Trust Regions", fontsize=fontsize, ha='left', rotation=rotation, va='top')

plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))  # Reducing x-ticks to at most 4
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=7))

# Adding labels and title
plt.xlabel("Policy Churn (%)", fontsize=14)
plt.ylabel("Performance (IQM)", fontsize=14)

plt.tick_params(axis='y', labelsize=14)
plt.tick_params(axis='x', labelsize=14)

# Display the plot
plt.show()
"""

"""
# Nstep vs performance

performance = [0.04, 0.214, 0.182, 0.145, 0.103]
#medians = [0.036, 0.108, 0.082, 0.067]
N = [1, 3, 7, 10, 20]

lower_bounds = [0.031, 0.199, 0.172, 0.136, 0.096]
upper_bounds = [0.051, 0.231, 0.202, 0.155, 0.113]

fontsize = 13
rotation = 15
N_label = [str(i) for i in N]

#plt.scatter(4, 0.218, color="red")
plt.errorbar(4, 0.218, yerr=0.017, fmt='o')
plt.text(4, 0.218, "10k", fontsize=fontsize, ha='left', rotation=rotation, va='bottom')

#plt.scatter(6, 0.257, color="green", marker="*")
plt.errorbar(6, 0.257, yerr=0.019, fmt='o')
plt.text(6, 0.257, "30k", fontsize=fontsize, ha='right', rotation=rotation, va='top')

#plt.scatter(8, 0.24, color="purple")
plt.errorbar(8, 0.24, yerr=0.019, fmt='o')
plt.text(8, 0.24, "50k", fontsize=fontsize, ha='left', rotation=rotation, va='bottom')

plt.plot(N, performance, label="IQM")
#plt.plot(N, medians, label="Median")
plt.xticks(N, N_label)

plt.xlabel("N-Step", fontsize=14)
plt.ylabel("Performance (IQM)", fontsize=14)
plt.fill_between(N, lower_bounds, upper_bounds, color='orange', alpha=0.1)

plt.tick_params(axis='y', labelsize=14)
plt.tick_params(axis='x', labelsize=14)

#plt.legend()

# Display the plot
plt.show()
"""

"""
# gamma vs performance

performance = [0.232, 0.264, 0.272, 0.214]
#medians = [0.183, 0.215, 0.192, 0.108]
N = [0.9, 0.95, 0.97, 0.99]
lower_bounds = [0.222, 0.25, 0.258, 0.199]
upper_bounds = [0.243, 0.278, 0.285, 0.231]

# Generate labels for your x-ticks
N_label = [str(i) for i in N]

# Plot the main lines
plt.plot(N, performance, label="IQM")

# Add the shaded area
plt.fill_between(N, lower_bounds, upper_bounds, color='blue', alpha=0.1)

# Setting x-ticks, labels and plot properties
plt.xticks(N, N_label)
plt.xlabel("Discount Rate", fontsize=14)
plt.ylabel("Performance (IQM)", fontsize=14)
plt.tick_params(axis='y', labelsize=14)
plt.tick_params(axis='x', labelsize=14)
#plt.legend(loc='lower left', fontsize=14)

# Display the plot
plt.show()
"""
fig, axs = plt.subplots(2, 1, figsize=(15, 10))
axs = axs.flatten()

# N vs policy churn
#atari
early = [12.7, 5.8, 4.1, 3.7, 3.3]
late = [6, 2.8, 2, 1.8, 1.6]

#procgen
#early = [16.4, 7, 5, 4.2]
#late = [8.8, 2.9, 2.3, 2.1]
N = [1, 3, 7, 10, 20]

N_label = [str(i) for i in N]

axs[0].plot(N, early, label="Early")
axs[0].plot(N, late, label="Late")
axs[0].set_xticks(N, N_label)

axs[0].scatter(6, 4, color="blue", marker="*")
axs[0].text(6, 4, "StableDQN", fontsize=13, ha='left', rotation=15, va='bottom')

axs[0].scatter(6, 1.8, color="orange", marker="*")
axs[0].text(6, 1.8, "StableDQN", fontsize=13, ha='left', rotation=15, va='bottom')

axs[0].set_xlabel("Value of N", fontsize=14)
axs[0].set_ylabel("Policy Churn (%)", fontsize=14)

axs[0].tick_params(axis='y', labelsize=14)
axs[0].tick_params(axis='x', labelsize=14)

axs[0].legend(fontsize=14)

# Display the plot
#plt.show()



# gamma vs policy churn

#atari
early = [3.6, 3.9, 4.4, 5.8]
late = [1.8, 2, 2.2, 2.8]

#procgen
#early = [3.6, 3.9, 4.4, 7]
#late = [1.8, 2, 2.2, 2.9]

gamma = [0.9, 0.95, 0.97, 0.99]


N_label = [str(i) for i in gamma]

axs[1].plot(gamma, early, label="Early")
axs[1].plot(gamma, late, label="Late")
axs[1].set_xticks(gamma, N_label)

axs[1].scatter(0.97, 4, color="blue", marker="*")
axs[1].text(0.97, 4, "StableDQN", fontsize=13, ha='left', rotation=15, va='bottom')

axs[1].scatter(0.97, 1.8, color="orange", marker="*")
axs[1].text(0.97, 1.8, "StableDQN", fontsize=13, ha='left', rotation=15, va='bottom')

#axs[1].gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=7))

axs[1].set_xlabel("Discount Rate", fontsize=14)
axs[1].set_ylabel("Policy Churn (%)", fontsize=14)

axs[1].tick_params(axis='y', labelsize=14)
axs[1].tick_params(axis='x', labelsize=14)

axs[1].legend(fontsize=14)

# Display the plot
plt.show()



"""# WallTime vs IQM

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

labels = [" StableDQN", " DER", " DrQ (eps)", "SPR"] #, "SR-SPR", "BBF"
iqms = [0.309,  0.183, 0.280,  0.337] #,  0.631, 1.045
walltimes = [8, 40, 62, 340] #, 430, 420
markers = ["*", None, None, None] #, None, None
sizes = [70, 30, 30, 30] #, 30, 30

fontsize = 14
rotation = 15

for i in range(len(labels)):
    if i == 1:
        ax.scatter(iqms[i], walltimes[i], marker=markers[i], s=sizes[i])
        ax.text(iqms[i], walltimes[i], labels[i], fontsize=fontsize, ha='left', rotation=rotation)
    else:
        if i == 0:
            ax.scatter(iqms[i], walltimes[i], marker=markers[i], s=sizes[i])
            ax.text(iqms[i], walltimes[i], labels[i], fontsize=fontsize, ha='right', rotation=rotation, va='bottom')
        else:
            ax.scatter(iqms[i], walltimes[i], marker=markers[i], s=sizes[i])
            ax.text(iqms[i], walltimes[i], labels[i], fontsize=fontsize, ha='right', rotation=rotation, va='top')


# Adding labels and title
ax.set_xlabel("Performance (IQM)", fontsize=14)
ax.set_ylabel("Walltime (GPU Minutes)", fontsize=14)

ax.set_yscale('log', base=2)
ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
#plt.xticks([20, 200, 500])

plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
matplotlib.rcParams.update({'font.size': 20})

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(16)  # Size here overrides font_prop

# Display the plot
plt.show()"""



"""
#Ablation graph

import matplotlib.pyplot as plt

# Data for DQN with Components Added
dqn_labels = ["DQN", "+Trust Regions", "Gamma=0.97", "+Annealing N"]
dqn_iqm = np.array([0.214, 0.232, 0.272, 0.257])

dqn_err = np.array([dqn_iqm - np.array([0.199, 0.22, 0.258, 0.246]), np.array([0.231, 0.245, 0.285, 0.284]) - dqn_iqm])
print(dqn_err.shape)

# Data for StableDQN with Components Removed (with switched order)
stabledqn_labels = ["StableDQN", "-Trust Regions", "Gamma=0.99", "-Annealing N"]
stabledqn_iqm = np.array([0.309, 0.28, 0.277, 0.286])

#lower_bounds_sta = [0.308, 0.272, 0.262, 0.275]
#upper_bounds_sta = [0.339, 0.289, 0.29, 0.298]

stabledqn_err = np.array([stabledqn_iqm - np.array([0.295, 0.272, 0.262, 0.275]), np.array([0.322, 0.289, 0.29, 0.298]) - stabledqn_iqm])
print(stabledqn_err.shape)

#stabledqn_err = np.array(upper_bounds_sta) - np.array(lower_bounds_sta)

# Creating side-by-side bar charts
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Adding only horizontal grid lines for better readability
axes[0].yaxis.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)  # Fainter gridlines
axes[1].yaxis.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)

# Plot for DQN with components added
axes[0].bar(dqn_labels, dqn_iqm, color='grey', yerr=dqn_err)
axes[0].set_title('DQN with Components Added')
axes[0].set_ylabel('IQM', fontsize=16)
axes[0].set_ylim([0.19, 0.35])
axes[0].tick_params(axis='x', rotation=45)

axes[0].tick_params(axis='x', labelsize=12, rotation=45)  # Larger x-axis tick labels
axes[0].tick_params(axis='y', labelsize=12)

#axes[0].set_xticklabels(dqn_labels, fontsize=14)
#axes[0].set_yticklabels(axes[0].get_yticks(), fontsize=14)

# Plot for StableDQN with components removed
axes[1].bar(stabledqn_labels, stabledqn_iqm, color='green', yerr=stabledqn_err)
axes[1].set_title('StableDQN with Components Removed')
axes[1].set_ylabel('IQM', fontsize=16)
axes[1].set_ylim([0.19, 0.35])
axes[1].tick_params(axis='x', rotation=45)

#axes[1].set_xticklabels(stabledqn_labels, fontsize=14)
#axes[1].set_yticklabels(axes[1].get_yticks(), fontsize=14)
axes[1].tick_params(axis='x', labelsize=12, rotation=45)  # Larger x-axis tick labels
axes[1].tick_params(axis='y', labelsize=12)

# Adjusting layout
plt.tight_layout()
plt.show()"""

"""
#batch size vs performance

batch_sizes = [8, 16, 32, 64]

stable = [0.292, 0.309, 0.292, 0.19]

# stable32 is called "Optim30k"
# stableDQN uses churn bug

stable_low = [0.274, 0.295, 0.273, 0.181]
stable_up = [0.312, 0.322, 0.31, 0.203]

dqn = [0.136, 0.202, 0.214, 0.18]

dqn_lower = [0.116, 0.185, 0.199, 0.165]
dqn_up = [0.155, 0.22, 0.231, 0.196]


plt.plot(batch_sizes, dqn, label="DQN")
plt.fill_between(batch_sizes, dqn_lower, dqn_up, color='blue', alpha=0.1)

plt.plot(batch_sizes, stable, label="StableDQN")
plt.fill_between(batch_sizes, stable_low, stable_up, color='orange', alpha=0.1)

N_label = [str(i) for i in batch_sizes]
# Setting x-ticks, labels and plot properties
plt.xticks(batch_sizes, N_label)
plt.xlabel("Batch Size", fontsize=14)
plt.ylabel("Performance (IQM)", fontsize=14)
plt.tick_params(axis='y', labelsize=14)
plt.tick_params(axis='x', labelsize=14)
plt.legend(loc='upper right', fontsize=14)

plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
matplotlib.rcParams.update({'font.size': 24})

# Display the plot
plt.show()
"""