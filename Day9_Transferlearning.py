import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models


# Image data load setup
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    'custom_data/train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    'custom_data/test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
base_model.trainable = False # Freeze the base model

num_classes = train_data.num_classes

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, epochs=5, validation_data=test_data)
model.save('custom_model9.h5')





