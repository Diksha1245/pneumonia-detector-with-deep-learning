import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image, image_dataset_from_directory

# Load dataset to extract class names
dataset_path = '/kaggle/input/labeled-chest-xray-images/chest_xray/train'  # Update with the path to your dataset

# Load the dataset (adjust parameters as needed)
dataset = image_dataset_from_directory(
    dataset_path,
    image_size=(256, 256),
    batch_size=32,
    label_mode='int'  # Use 'int' for integer labels or 'categorical' for one-hot encoded labels
)

# Extract class names from the dataset
class_names = dataset.class_names

print("Class names:", class_names)

# Function to preprocess an image and make a prediction
def predict_image(image_path, model, class_names, threshold=0.5):
    # Load the image and preprocess it
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match model input shape
    img_array /= 255.0  # Normalize pixel values (assuming your model expects normalized inputs)

    # Make prediction
    prediction = model.predict(img_array)
    print(prediction)

    # Determine the class index based on the threshold
    class_index = 1 if prediction >= threshold else 0

    # Get the class name from the class index
    class_name = class_names[class_index]

    # Print the prediction and the corresponding class
    print(f"Prediction: {prediction[0][0]:.4f}")
    print(f"Predicted class: {class_name}")

# Predict on new images
predict_image('/kaggle/input/labeled-chest-xray-images/chest_xray/test/NORMAL/NORMAL-1858497-0001.jpeg', model, class_names)
predict_image('/kaggle/input/labeled-chest-xray-images/chest_xray/test/NORMAL/NORMAL-2123652-0001.jpeg', model, class_names)
predict_image('/kaggle/input/labeled-chest-xray-images/chest_xray/test/PNEUMONIA/BACTERIA-1351146-0002.jpeg', model, class_names)
predict_image('/kaggle/input/labeled-chest-xray-images/chest_xray/test/PNEUMONIA/BACTERIA-1351146-0002.jpeg', model, class_names)
predict_image('/kaggle/input/labeled-chest-xray-images/chest_xray/train/NORMAL/NORMAL-1140710-0001.jpeg', model, class_names)
predict_image('/kaggle/input/labeled-chest-xray-images/chest_xray/train/PNEUMONIA/BACTERIA-1076722-0001.jpeg', model, class_names)

