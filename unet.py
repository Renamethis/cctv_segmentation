import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import os
import cv2
from glob import glob
import matplotlib.pyplot as plt

train_img = "images/leftImg8bit/train/**/*.png"
train_annotations = "gtFine/train/**/*_gtFine_labelIds.png"
val_img = "images/leftImg8bit/val/**/*.png"
val_annotations = "gtFine/val/**/*_gtFine_labelIds.png"
test_img = "images/leftImg8bit/val/**/*.png"
test_annotations = "gtFine/test/**/*_gtFine_labelIds.png"


COLOR_MAP = { # TODO: Fix color map if needed
    (0, 0, 0): 0,       # unlabeled
    (0, 0, 1): 1,       # ego vehicle
    (0, 0, 2): 2,       # rectification border
    (0, 0, 3): 3,       # out of roi
    (0, 0, 4): 4,       # static
    (0, 0, 5): 5,
    (81, 0, 81): 6,     # ground
    (128, 64, 128): 7,  # road
    (244, 35, 232): 8,  # sidewalk
    (250, 170, 160): 9, # parking
    (230, 150, 140): 10,# rail track
    (70, 70, 70): 11,   # building
    (102, 102, 156): 12,# wall
    (190, 153, 153): 13,# fence
    (180, 165, 180): 14,# guard rail
    (150, 100, 100): 15,# bridge
    (150, 120, 90): 16, # tunnel
    (153, 153, 153): 17,# pole
    (153, 153, 153): 18,# polegroup
    (250, 170, 30): 19, # traffic light
    (220, 220, 0): 20,  # traffic sign
    (107, 142, 35): 21, # vegetation
    (152, 251, 152): 22,# terrain
    (70, 130, 180): 23, # sky
    (220, 20, 60): 24,  # person
    (255, 0, 0): 25,    # rider
    (0, 0, 142): 26,    # car
    (0, 0, 70): 27,     # truck
    (0, 60, 100): 28,   # bus
    (0, 0, 90): 29,     # caravan
    (0, 0, 110): 30,    # trailer
    (0, 80, 100): 31,   # train
    (0, 0, 230): 32,    # motorcycle
    (119, 11, 32): 33,   # bicycle
    (0, 0, 6): 34,       # static
}
IMG_HEIGHT, IMG_WIDTH = 128, 128  # Adjust as necessary
BATCH_SIZE = 8

NUM_CLASSES = len(COLOR_MAP)  # Number of classes in the dataset
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
def rgb_to_class_id(rgb_label):
    """Convert RGB mask to class ID mask using the color map."""
    class_id_label = np.zeros(rgb_label.shape[:2], dtype=np.int32)
    for rgb, class_id in COLOR_MAP.items():
        mask = np.all(rgb_label == rgb, axis=-1)
        class_id_label[mask] = class_id
    return class_id_label

def load_image(path, resize_shape=(IMG_HEIGHT, IMG_WIDTH)):
    img = cv2.imread(path, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, resize_shape)
    img = img / 255.0  # Normalize image
    return img

def load_mask(path, resize_shape=(IMG_HEIGHT, IMG_WIDTH)):
    color_mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    color_mask = cv2.resize(color_mask, resize_shape)
    return color_mask

def unet_model(input_shape):
    """Build U-Net model."""
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(len(COLOR_MAP), (1, 1), activation='softmax')(c9)  # Output layer

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Create U-Net model
input_shape = (128, 128, 3)
model = unet_model(input_shape)

train_input = np.array([load_image(img) for img in sorted(glob(train_img, recursive=True))])
train_labels = np.array([load_mask(mask) for mask in sorted(glob(train_annotations, recursive=True))])
val_input = np.array([load_image(img) for img in sorted(glob(val_img, recursive=True))])
val_labels = np.array([load_mask(mask) for mask in sorted(glob(val_annotations, recursive=True))])
test_input = np.array([load_image(img) for img in sorted(glob(test_img, recursive=True))])
test_labels = np.array([load_mask(mask) for mask in sorted(glob(test_annotations, recursive=True))])

# Convert COLOR_MAP to a list or array for easy mapping
COLOR_MAP_LIST = np.array([rgb for rgb, class_id in sorted(COLOR_MAP.items(), key=lambda x: x[1])], dtype=np.uint8)

def class_id_to_color(class_id_mask):
    """Convert class ID mask to color-coded RGB mask."""
    # Create an empty color mask
    color_mask = np.zeros((class_id_mask.shape[0], class_id_mask.shape[1], 3), dtype=np.uint8)
    
    # Map each class ID to its corresponding color
    for class_id, color in COLOR_MAP.items():
        color_mask[class_id_mask == class_id] = color
        
    return color_mask

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training
history = model.fit(
    train_input, train_labels,
    validation_data=(val_input, val_labels),
    batch_size=8,
    epochs=140,
    verbose=1
)


# Convert COLOR_MAP to a list or array for easy mapping
COLOR_MAP_LIST = np.array([rgb for rgb, class_id in sorted(COLOR_MAP.items(), key=lambda x: x[1])], dtype=np.uint8)

def class_id_to_color(class_id_mask):
    """Convert class ID mask to color-coded RGB mask."""
    # Create an empty color mask
    color_mask = np.zeros((class_id_mask.shape[0], class_id_mask.shape[1], 3), dtype=np.uint8)
    
    # Map each class ID to its corresponding color
    for class_id, color in COLOR_MAP.items():
        color_mask[class_id_mask == class_id] = color
        
    return color_mask

def plot_training_history(history):
    """Plot training and validation loss/accuracy."""
    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

plot_training_history(history)

def visualize_predictions(model, images, labels):
    preds = model.predict(images)
    preds = np.argmax(preds, axis=-1)
    # Visualize the first image in the 1batch and its corresponding label
    cnt = 0
    for prediction in preds:
        plt.figure(figsize=(12, 6))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(images[cnt])
        plt.axis('off')

        # Ground truth
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(labels[cnt])  # Convert label to color
        plt.axis('off')

        # Model predictions
        plt.subplot(1, 3, 3)
        plt.title("Model Predictions")
        plt.imshow(prediction) # Show the first prediction
        plt.axis('off')
        cnt += 1

    plt.show()

# Visualize predictions
visualize_predictions(model, test_input[:10], test_labels[:10])
