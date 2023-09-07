# Importing required libraries
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Ground-truth labels and predicted labels
y_true = [0, 1, 0, 1, 0, 1, 1, 0]
y_pred = [0, 1, 1, 0, 0, 1, 1, 0]

# Calculate the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Label names for the confusion matrix
class_names = ["Positive", "Negative"]

# Plotting the confusion matrix
# Create the plot
fig, ax = plt.subplots()

# Display the confusion matrix
cax = ax.matshow(cm, cmap=plt.cm.Blues)

# Show color bar
plt.colorbar(cax)

# Add text annotations
# Interpreting the confusion matrix
# True Negative (TN) = cm[0,0]
# False Positive (FP) = cm[0,1]
# False Negative (FN) = cm[1,0]
# True Positive (TP) = cm[1,1]
plt.text(0, 0, str(cm[1][1]), va="center", ha="center")
plt.text(0, 1, str(cm[0][1]), va="center", ha="center")
plt.text(1, 1, str(cm[0][0]), va="center", ha="center")
plt.text(1, 0, str(cm[1][0]), va="center", ha="center")

# Add labels and title
plt.xticks(np.arange(len(class_names)), class_names)
plt.yticks(np.arange(len(class_names)), class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig(f"confusion_matrix.png")

# Accuracy = (TN + TP) / (TN + FP + FN + TP)
# Precision = TP / (TP + FP)
# Recall = TP / (TP + FN)

print(f"True Negatives: {cm[0, 0]}")
print(f"False Positives: {cm[0, 1]}")
print(f"False Negatives: {cm[1, 0]}")
print(f"True Positives: {cm[1, 1]}")

accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
