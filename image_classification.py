import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create an ImageDataGenerator to preprocess images
datagen = ImageDataGenerator(rescale=1./255)

# Load training and testing data
train_generator = datagen.flow_from_directory(
    'path/to/train/directory',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = datagen.flow_from_directory(
    'path/to/validation/directory',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Convolutional Neural Network
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# Evaluate
test_loss, test_acc = model.evaluate(validation_generator)
print('Test accuracy:', test_acc)

# Load a new image
new_image = load_img('path/to/new/image.jpg', target_size=(150, 150))
new_image = img_to_array(new_image)
new_image = np.expand_dims(new_image, axis=0)

# Make prediction
predicted_class = model.predict(new_image)
predicted_label = class_labels[np.argmax(predicted_class)]
print('Predicted class:', predicted_label)