
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Define paths
train_images_dir = 'C:/Users/dslab/Downloads/train-20240522T054636Z-001/train/images'
labels_dir = 'C:/Users/dslab/Downloads/train-20240522T054636Z-001/train/labels'

# Function to create a dictionary of image paths and their corresponding labels
def create_labels_dict(labels_dir):
    labels_dict = {}
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):  # Assuming label files are .txt
            file_path = os.path.join(labels_dir, label_file)
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])  # Assuming first part is the class_id
                    image_name = os.path.splitext(label_file)[0] + '.jpg'  # Assuming image extension is .jpg
                    labels_dict[image_name] = class_id
    return labels_dict

# Create labels dictionary
labels_dict = create_labels_dict(labels_dir)

# Get unique labels
unique_labels = list(set(labels_dict.values()))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}

# Image data generator for preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Use 20% of data for validation
)

# Helper function to load images and labels
def load_data(image_dir, labels_dict, target_size=(224, 224)):
    images = []
    labels = []
    for image_name, class_id in labels_dict.items():
        image_path = os.path.join(image_dir, image_name)
        if os.path.exists(image_path):
            img = load_img(image_path, target_size=target_size)
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(class_id)
    return np.array(images), np.array(labels)

# Load images and labels
images, labels = load_data(train_images_dir, labels_dict)

# Split into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(unique_labels))
y_val = tf.keras.utils.to_categorical(y_val, num_classes=len(unique_labels))

# Define the model
# Load pre-trained MobileNetV2 model without the top layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(unique_labels), activation='softmax')(x)

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Image data generator for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Image data generator for validation
val_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
validation_generator = val_datagen.flow(x_val, y_val, batch_size=32)

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

# Making Predictions
from tensorflow.keras.preprocessing import image

# Define class names
class_names = [
    "bibimbap", "bulgogi", "godeungeogui", "jjambbong", "ramyun",
    "yangnyumchicken", "duinjangjjigae", "gamjatang", "gimbap", "jeyukbokkeum",
    "jjajangmyeon", "kalguksu", "kimchijjigae", "mandu", "pajeon",
    "samgyetang", "samgyeopsal", "sundaegukbap", "tteokbokki", "tteokguk"
]

