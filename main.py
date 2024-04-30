import tensorflow as tf
import os
import numpy as np
import scipy

# from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

IMG_SIZE = 160
batch_size = 32


data_dir = "split_data"

# train_ds = tf.keras.utils.image_dataset_from_directory(
#     data_dir,
#     validation_split=0.2,
#     subset="training",
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=batch_size,
#     seed=124,
#     labels="inferred",
# )

# val_ds = tf.keras.utils.image_dataset_from_directory(
#     data_dir,
#     validation_split=0.2,
#     subset="validation",
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=batch_size,
#     seed=124,
#     labels="inferred",
# )


# for validation we can use this method
# validation_generator = test_datagen.flow_from_directory(
#'validation_dir',  # This should be your validation directory path
# target_size=(150, 150),
# batch_size=20,
# class_mode="categorical"  # Use 'categorical' for multi-class classification
# )
# this will require us to create the actual validation data set

# history = model.fit(
#     train_ds,
#     epochs=5,
#     validation_data=val_ds  # Pass the validation dataset for validation during training
# )

# import matplotlib.pyplot as plt

# # Plot training & validation accuracy values
# plt.plot(history.history['accuracy'], label='Training accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation accuracy')
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(loc='upper left')
# plt.show()

# # Plot training & validation loss values
# plt.plot(history.history['loss'], label='Training loss')
# plt.plot(history.history['val_loss'], label='Validation loss')
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(loc='upper left')
# plt.show()


# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    # rotation_range=40,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2,
    # vertical_flip=True,
    # fill_mode="nearest",
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
valid_datagen = ImageDataGenerator(rescale=1.0 / 255)


# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    data_dir + "/train",  # This is the source directory for training images
    target_size=(IMG_SIZE, IMG_SIZE),  # All images will be resized to 150x150
    batch_size=batch_size,
    class_mode="categorical",
)

test_datagen = test_datagen.flow_from_directory(
    data_dir + "/test",  # This is the source directory for training images
    target_size=(IMG_SIZE, IMG_SIZE),  # All images will be resized to 150x150
    batch_size=batch_size,
    class_mode="categorical",
)

valid_datagen = valid_datagen.flow_from_directory(
    data_dir + "/validation",  # This is the source directory for training images
    target_size=(IMG_SIZE, IMG_SIZE),  # All images will be resized to 150x150
    batch_size=batch_size,
    class_mode="categorical",
)

num_classes = 6

# Load pre-trained MobileNetV2 model without top classification layer
base_model = MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
)


# # Freeze pre-trained layers
# for layer in base_model.layers:
#     layer.trainable = False

model = tf.keras.Sequential(
    [
        # base_model,
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        # tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        # tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),  # Adding Dropout
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(train_generator, epochs=14, validation_data=valid_datagen)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_datagen)

# Print the test loss and accuracy
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

model.summary()

plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0.5, 1])
plt.legend(loc="lower right")

plt.show()
