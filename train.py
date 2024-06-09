import tensorflow as tf
import os
import random
import numpy as np
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from Unet import model
from utils import train_images, train_masks, test_images, IMG_WIDTH, IMG_HEIGHT

seed = 42
np.random.seed = seed


checkpointer = tf.keras.callbacks.ModelCheckpoint('model.h5', verbose=1, save_best_only=True)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

results = model.fit(train_images, train_masks, validation_split=0.1, batch_size=24, epochs=50, callbacks=callbacks)

# Predict on test data
preds_test = model.predict(test_images, verbose=1)

# Resize back to original size
preds_test_resized = []
for i in range(len(preds_test)):
    preds_test_resized.append(resize(np.squeeze(preds_test[i]), 
                                     (IMG_WIDTH, IMG_HEIGHT), 
                                     mode='constant', preserve_range=True))

# Plot a test image and its predicted mask
imshow(test_images[0])
plt.show()
imshow(np.squeeze(preds_test[0]))
plt.show()

# Calculate IoU
def iou_metric(y_true, y_pred):
    y_true = y_true.astype(np.bool)
    y_pred = y_pred > 0.5
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


