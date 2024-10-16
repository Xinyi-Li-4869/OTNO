import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree
from scipy.spatial import cKDTree
import ot

# Define points on a straight line
x_line = np.linspace(0, 10, 20)
y_line = np.zeros_like(x_line) + 1

# Define points on a curve (e.g., a parabola)
x_curve = np.linspace(0, 10, 20)
y_curve = - (x_curve - 5)**2 / 10 - 1

line_color = '#a00000'  # medium blue 00b0be e25759
curve_color = '#298c8c'  # blue 59a89c
connection_color = '#595959'  # green

# Prepare figure
fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))

# Nearest Neighbor Search using KDTree
tree = KDTree(np.column_stack([x_curve, y_curve]))
_, idx = tree.query(np.column_stack([x_line, y_line]))
axs[0].scatter(x_line, y_line, color=line_color, label='Line')
axs[0].scatter(x_curve, y_curve, color=curve_color, label='Curve')
for i in range(len(x_line)):
    axs[0].plot([x_line[i], x_curve[idx[i]]], [y_line[i], y_curve[idx[i]]], color=connection_color, linestyle='--')
axs[0].set_title('Nearest Neighbor Search')
axs[0].axis('equal')
axs[0].axis('off')
'''# Ball Tree Search
ball_tree = BallTree(np.column_stack([x_curve, y_curve]))
_, idx_bt = ball_tree.query(np.column_stack([x_line, y_line]))
idx_bt = idx_bt.flatten()  # Flattening the idx array for proper indexing

axs[1].plot(x_line, y_line, 'ro', label='Line')
axs[1].plot(x_curve, y_curve, 'bo', label='Curve')
for i in range(len(x_line)):
    axs[1].plot([x_line[i], x_curve[idx_bt[i]]], [y_line[i], y_curve[idx_bt[i]]], 'k--')
axs[1].set_title('Ball Tree Search')
axs[1].axis('equal')'''
# BallTree for Radius Search
line_points = np.column_stack([x_line, y_line])
curve_points = np.column_stack([x_curve, y_curve])
# Create a cKDTree with curve points
curve_tree = cKDTree(curve_points)
# Radius for search
radius = 2.5  # Define the radius for the search
# Query the cKDTree for neighbors within radius
indices_radius = curve_tree.query_ball_point(line_points, radius)
# Ball Search Visualization
axs[1].scatter(x_line, y_line, color=line_color, label='Line')
axs[1].scatter(x_curve, y_curve, color=curve_color, label='Curve')
for i, indices in enumerate(indices_radius):
    for index in indices:
        axs[1].plot([x_line[i], x_curve[index]], [y_line[i], y_curve[index]], color=connection_color, linestyle='--')
axs[1].set_title('Ball Search within Radius')
axs[1].axis('equal')
axs[1].axis('off')

# Optimal Transport Plan
M = ot.dist(np.column_stack([x_line, y_line]), np.column_stack([x_curve, y_curve]), metric='sqeuclidean')
a = np.ones(len(x_line)) / len(x_line)
b = np.ones(len(x_curve)) / len(x_curve)
transport_plan = ot.emd(a, b, M)
axs[2].scatter(x_line, y_line, color=line_color, label='Line')
axs[2].scatter(x_curve, y_curve, color=curve_color, label='Curve')
for i in range(len(x_line)):
    for j in range(len(x_curve)):
        if transport_plan[i, j] > 1e-5:
            axs[2].plot([x_line[i], x_curve[j]], [y_line[i], y_curve[j]], color=connection_color, linestyle='--')#, alpha=transport_plan[i, j] * 5)
axs[2].set_title('Optimal Transport Plan')
axs[2].axis('equal')
axs[2].axis('off')

# Show plots
plt.tight_layout()
plt.legend()
plt.show()

plt.savefig('/central/groups/tensorlab/xinyi/OT/search_comparison.png')