import numpy as np

speed_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
max_speeds = speed_matrix.max()
print(max_speeds)
max_speed = max_speeds.max()
print(max_speed)  # 输出: 9
