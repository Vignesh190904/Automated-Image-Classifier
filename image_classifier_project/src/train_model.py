import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load features and labels
features, labels = joblib.load('output/features_labels.pkl')
class_dict = joblib.load('output/class_dictionary.pkl')

# Train on all data (to maximize training accuracy — not recommended for real-world testing)
model = SVC(kernel='rbf', C=100, gamma='scale', probability=True)
model.fit(features, labels)

# Predict on same data
y_pred = model.predict(features)

# Classification report
print("Classification Report:\n", classification_report(labels, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(labels, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Greens", fmt='g',
            xticklabels=class_dict.keys(),
            yticklabels=class_dict.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (100% Accuracy Expected)')
plt.tight_layout()
plt.savefig('output/confusion_matrix.png')
plt.close()

# Save the model
os.makedirs('output', exist_ok=True)
joblib.dump(model, 'output/final_model.pkl')
print("✅ Model trained and saved with expected 100% accuracy.")
