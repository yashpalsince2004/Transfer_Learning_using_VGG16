# 📦 Transfer Learning with VGG16 for Custom Image Classification

This project demonstrates **transfer learning** using the **VGG16** convolutional neural network to classify images from a custom dataset. The model is trained on your dataset by leveraging the pre-trained weights of VGG16 on ImageNet, freezing the base layers, and adding a custom classifier on top.

---

## 🚀 Features

✅ Uses **VGG16** as the base model  
✅ Freezes the pre-trained layers for efficiency  
✅ Adds a fully connected classifier for custom classes  
✅ Trains on images from a user-provided dataset  
✅ Saves the trained model in `.h5` format  
✅ Simple and modular Keras implementation

---

## 🗂 Dataset Structure

Your dataset should be organized in the following directory structure:

```
custom_data/
│
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
│
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
```

Each class folder contains the images belonging to that class.

---

## 📝 Requirements

- Python ≥ 3.7  
- TensorFlow ≥ 2.x  
- GPU recommended but not mandatory

You can install the dependencies via:

```bash
pip install tensorflow
```

---

## 📄 How it Works

1️⃣ Loads training and testing images using `ImageDataGenerator` with normalization.  
2️⃣ Loads the **VGG16** model without the top layers (`include_top=False`) and freezes its weights.  
3️⃣ Adds a custom head:
   - `Flatten` layer
   - Dense layer with ReLU
   - Output layer with Softmax for multi-class classification
4️⃣ Compiles the model with:
   - Optimizer: Adam
   - Loss: Categorical Crossentropy
   - Metric: Accuracy
5️⃣ Trains the model for 5 epochs and validates it.  
6️⃣ Saves the trained model as `custom_model9.h5`.

---

## 📜 Code Overview

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Data Preparation
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory('custom_data/train', target_size=(64, 64), batch_size=32, class_mode='categorical')
test_data = test_datagen.flow_from_directory('custom_data/test', target_size=(64, 64), batch_size=32, class_mode='categorical')

# Model Building
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(train_data, epochs=5, validation_data=test_data)

# Save
model.save('custom_model9.h5')
```

---

## 📊 Results

At the end of training, the model is capable of classifying images into the custom-defined classes with reasonable accuracy even with just 5 epochs, thanks to transfer learning.

---

## 🧪 To Run

1. Clone this repository:
   ```bash
   git clone <https://github.com/yashpalsince2004/Transfer_Learning_using_VGG16.gitl>
   cd <custom_data>
   ```
2. Make sure you have prepared the dataset as shown above.  
3. Run the training script:
   ```bash
   python Day9_Transferlearning.py
   ```

The trained model will be saved as `custom_model9.h5`.

---

## 🙋‍♂️ Author

[Yash Pal](https://www.linkedin.com/in/yash-pal-since2004)

---

## 🌟 Notes

- You can unfreeze some of the VGG16 layers if you wish to fine-tune the base model.
- Increase the number of epochs or batch size depending on your hardware and dataset size.
- Replace VGG16 with other models (like ResNet50, InceptionV3) by modifying the base model part.

---

## 📄 License

This project is provided for educational purposes and is open-source. Please cite the author if you use it in a publication.
