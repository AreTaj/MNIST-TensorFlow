import tensorflow as tf
import tensorflow_datasets as tfds

# Load 'MNIST' dataset
(mnist_train, mnist_test), info = tfds.load('mnist', split=['train', 'test'], with_info=True)
 
#print("Features:\n",info.features)  # Get information about the dataset features
    # 'image': Image(shape=(28, 28, 1), dtype=uint8),
    # 'label': ClassLabel(shape=(), dtype=int64, num_classes=10)

print(info)
