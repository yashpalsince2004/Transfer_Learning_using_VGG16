import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import numpy as np
import seaborn as sns

model = load_model('custom_model9.h5')
print("✅ Model loaded successfully!")

# Image data load setup
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    'custom_data/test',
    target_size=(64, 64),  # Same size as training
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

predictions = model.predict(test_data, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_data.classes

accuracy = np.sum(predicted_classes == true_classes) / len(true_classes)
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")

import matplotlib.pyplot as plt

labels = list(test_data.class_indices.keys())  # ['cat', 'dog', ...]
test_data.reset()

plt.figure(figsize=(15, 8))
for i in range(10):
    img, label = next(test_data)
    true_label = np.argmax(label[0])
    pred_label = predicted_classes[i]
    
    plt.subplot(2, 5, i + 1)
    plt.imshow(img[0])
    color = 'green' if true_label == pred_label else 'red'
    plt.title(f"True: {labels[true_label]}\nPred: {labels[pred_label]}", color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(true_classes, predicted_classes, target_names=labels))

cm = confusion_matrix(true_classes, predicted_classes)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


