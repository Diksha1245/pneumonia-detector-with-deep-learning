import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Sequential,Model
import numpy as np
from sklearn.utils import class_weight

# Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '/kaggle/input/labeled-chest-xray-images/chest_xray/train',
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    '/kaggle/input/labeled-chest-xray-images/chest_xray/test',
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary'
)

# Calculate class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes

)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model.keras',
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

# Transfer Learning with VGG16
local_weights_path = "/kaggle/input/vgg16-weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
base_model = VGG16(weights=local_weights_path, include_top=False, input_shape=(256, 256, 3))

for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

"""model = Sequential([
    Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu", input_shape=(256,256,3)),
    MaxPooling2D(pool_size=2, strides=2, padding="valid"),
    BatchNormalization(),

    Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"),
    MaxPooling2D(pool_size=2, strides=2, padding="valid"),
    BatchNormalization(),
    Conv2D(filters=100, kernel_size=(3,3), padding="same", activation="relu"),
    MaxPooling2D(pool_size=2, strides=2, padding="valid"),
    BatchNormalization(),

    Flatten(),
    Dense(256, activation="relu"),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),

    Dropout(0.5),  # Added dropout layer
    Dense(1, activation="sigmoid")
])"""


model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    class_weight=dict(enumerate(class_weights)),
    callbacks = [checkpoint]
)

# Evaluate the model
test_loss, test_acc = model.evaluate(validation_generator, steps=validation_generator.samples // validation_generator.batch_size)
print('Test accuracy:', test_acc)
