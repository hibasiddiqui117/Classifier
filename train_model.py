import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Rescaling, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set paths and parameters
img_height, img_width = 180, 180
batch_size = 32  # You can try 16 or 64 based on your system
train_path = 'dataset/train'
val_path = 'dataset/validation'

# Check if GPU is available
print("GPU Available:", "Yes" if tf.config.list_physical_devices('GPU') else "No")

# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_path,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Model building with improved architecture
model = Sequential([
    Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    Conv2D(32, 3, activation='relu'),  # Increased filters
    MaxPooling2D(),
    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(),
    Conv2D(128, 3, activation='relu'),  # Added more layers
    MaxPooling2D(),
    Flatten(),
    Dropout(0.5),  # Increased dropout for better regularization
    Dense(256, activation='relu'),  # Bigger dense layer
    Dense(len(train_ds.class_names), activation='softmax')
])

# Compile with adjusted learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks for better training
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
]

# Train with callbacks
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,  # Increased epochs since we have early stopping
    callbacks=callbacks
)

# Save model
model.save('Image_classify.keras')

print(f"✅ Model training complete! Saved as 'Image_classify.keras'")
print(f"Final Validation Accuracy: {max(history.history['val_accuracy']):.2%}")