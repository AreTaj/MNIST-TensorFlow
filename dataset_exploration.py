import tensorflow as tf
import tensorflow_datasets as tfds

# Load 'MNIST' dataset
(ds_train, ds_test),info = tfds.load('mnist', split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=True)

""" Section: Dataset Features and Information """
# Output to .txt file
with open('exploratory_output.txt', 'w') as f:
    # Get information about the dataset features
    print("Features:\n",info.features,"\n", file=f) 
    """
    Features:
    'image': Image(shape=(28, 28, 1), dtype=uint8),
    'label': ClassLabel(shape=(), dtype=int64, num_classes=10)
    """

    print(info, file=f)

""" Section: Data Visualization """
import numpy as np
import matplotlib.pyplot as plt

num_samples = 10

plt.figure(figsize=(10, 5))
count = 0
for image, label in ds_train:
  if count < num_samples:
    plt.subplot(2, 5, count + 1)
    plt.imshow(image.numpy().squeeze(), cmap='gray')
    plt.title(f"Label: {label.numpy()}")
    plt.axis('off')
    count += 1
  else:
    break
""" for i, (image, label) in enumerate(ds_train.take(num_samples)):
    plt.subplot(2, 5, i + 1)
    plt.imshow(image.numpy().squeeze(), cmap='gray')
    plt.title(f"Label: {label.numpy()}")
    #plt.show() """
plt.savefig('mnist_sample_images.png')