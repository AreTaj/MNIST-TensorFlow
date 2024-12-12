import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # Import for image processing
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load training and testing data (using the first call)
(mnist_train, mnist_test), info = tfds.load('mnist', split=['train', 'test'], with_info=True)

train_data = mnist_train.map(lambda x: (tf.cast(x['image'], tf.float32) / 255.0, x['label']))
test_data = mnist_test.map(lambda x: (tf.cast(x['image'], tf.float32) / 255.0, x['label']))

num_classes = info.features['label'].num_classes  # 10 for MNIST dataset

# One-hot encode labels
train_labels = train_data.map(lambda x, y: (x, to_categorical(y, num_classes=num_classes)))
test_labels = test_data.map(lambda x, y: (x, to_categorical(y, num_classes=num_classes)))

train_batches = train_data.shuffle(1000).batch(32).map(lambda x, y: (x, y))
test_batches = test_data.batch(32).map(lambda x, y: (x, y))

num_classes = info.features['label'].num_classes  # 10 for MNIST dataset

# Convolutional Neural Network
model = Sequential([
    Conv2D(32, (3, 3), activation='relu'),
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
    train_batches,
    steps_per_epoch=len(train_batches),
    epochs=10,
    validation_data=test_batches,
    validation_steps=len(test_batches)
)

# Evaluate
test_loss, test_acc = model.evaluate(test_batches)
print('Test accuracy:', test_acc)

# Load a new image (assuming it's a grayscale image)
new_image = load_img('path/to/new/image.jpg', target_size=(28, 28), grayscale=True)  # Adjust grayscale if needed
new_image = img_to_array(new_image)
new_image = np.expand_dims(new_image, axis=0)

# Make prediction
predicted_class = model.predict(new_image)
predicted_label = np.argmax(predicted_class)  # Assuming class labels are integers (0-9)
print('Predicted class:', predicted_label)