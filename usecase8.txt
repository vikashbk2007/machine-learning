import numpy as np
from collections import Counter

# ----------------------------
# Training Data
# ----------------------------
X = np.array([
    [2, 30],
    [3, 35],
    [5, 60],
    [6, 65]
], dtype=float)

y = np.array(["Fail", "Fail", "Pass", "Pass"])

# ----------------------------
# Test Data
# ----------------------------
test_point = np.array([4, 50], dtype=float)

# K value
k = 3

# ----------------------------
# Step 1: Calculate Euclidean Distance
# ----------------------------
distances = np.linalg.norm(X - test_point, axis=1)

# ----------------------------
# Step 2: Get K nearest neighbors
# ----------------------------
k_indices = np.argsort(distances)[:k]
k_nearest_labels = y[k_indices]

# ----------------------------
# Step 3: Majority Voting
# ----------------------------
vote_count = Counter(k_nearest_labels)
prediction = vote_count.most_common(1)[0][0]

# ----------------------------
# Output
# ----------------------------
print("Distances:", np.round(distances, 2))
print("K Nearest Labels:", k_nearest_labels)
print("Vote Count:", vote_count)
print("Predicted Result:", prediction)
