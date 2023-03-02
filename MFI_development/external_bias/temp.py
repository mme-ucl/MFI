import numpy as np

# Define your original grid as a NumPy array from -6 to 6 with 401 points
original_grid = np.linspace(-6, 6, num=41)

# Define the range and spacing for the new values
new_range = 2  # the range will be from -8 to 8
new_spacing = original_grid[1] - original_grid[0]  # use the same spacing as the original array

# Compute the number of new values to add on each side
num_new_values = int(new_range / new_spacing)+1

end_values = (-6 - new_spacing* num_new_values, 6 + new_spacing* num_new_values)

# Use numpy.pad() to add padding to the original grid
padded_grid = np.pad(original_grid, (num_new_values, num_new_values), mode='linear_ramp', end_values=end_values)

# Print the original grid and the padded grid to compare them
print("Original grid:\n", original_grid)
print("\nPadded grid:\n", padded_grid)

