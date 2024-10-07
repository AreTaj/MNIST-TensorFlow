import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load training and testing data
mnist_data = tfds.load("mnist")
mnist_train, mnist_test = mnist_data["train"], mnist_data["test"]   # Use pre-divided dataset
(mnist_train, mnist_test), info = tfds.load('mnist', split=['train', 'test'], with_info=True)
assert isinstance(mnist_train, tf.data.Dataset)     # Verify that mnist_train is an instance of the tf.data.Dataset class

""" # OPTIONAL: Normalize pixel values to the range [0, 1]
train_data = mnist_train.map(lambda x: (tf.cast(x['image'], tf.float32) / 255.0, x['label']))
test_data = mnist_test.map(lambda x: (tf.cast(x['image'], tf.float32) / 255.0, x['label'])) """

train_batches = mnist_train.shuffle(1000).batch(32)
test_batches = mnist_test.batch(32)

num_classes = info.features['label'].num_classes       # 10 for MNIST dataset

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

# Load a new image
new_image = load_img('path/to/new/image.jpg', target_size=(150, 150))
new_image = img_to_array(new_image)
new_image = np.expand_dims(new_image, axis=0)

# Make prediction
predicted_class = model.predict(new_image)
predicted_label = class_labels[np.argmax(predicted_class)]
print('Predicted class:', predicted_label)