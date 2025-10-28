# Plot the Brodmann areas for colored legends.

# %%
from util.easy_import import *


bas = [1, 4, 7, 40, 39, 19]
colors = ['blue', 'green', 'black', 'cyan', 'magenta', 'yellow']
names = [f'#{e}' for e in bas]

# %%
# Plot lines as legend
# I only want the legend for image notion.
# Create a figure specifically for the legend
fig, ax = plt.subplots(figsize=(8, 3))

# Create invisible lines just for the legend
lines = []
for color in colors:
    line = plt.Line2D([0], [0], color=color, linewidth=3)
    lines.append(line)

# Create the legend
ax.legend(lines, names, loc='center', frameon=False, ncol=1)

# Remove axes and set tight layout
ax.axis('off')
plt.tight_layout()
plt.show()

# %%
