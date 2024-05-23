# After training the model, load the trained model weights if needed
# final_cnn.load_weights("/content/drive/My Drive/1_LiveProjects/Project5_Age_Detection/age_input_output/output/cnn_logs/age_model_checkpoint.h5")

# Function to predict whether an image corresponds to a child or not
import tensorflow as tf
def predict_child_or_not(image_path):
    # Load and preprocess the image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=1)  # Convert to grayscale if needed
    image = tf.image.resize(image, [200, 200])      # Resize as per model input shape
    image = tf.expand_dims(image, axis=0)            # Add batch dimension

    # Make predictions using the trained model
    from keras.models import load_model
    final_cnn=load_model('Age_Model_Acc_0.825.h5')
    predictions = final_cnn.predict(image)
    
    # Determine whether the image corresponds to a child or not based on the probabilities
    child_probability = predictions[:, :3].sum()     # Sum of probabilities for classes 0-2 (child)
    not_child_probability = predictions[:, 3:].sum() # Sum of probabilities for classes 3-6 (not child)

    # Apply threshold
    threshold = 0.5
    if child_probability > threshold * (child_probability + not_child_probability):
        return "Child"
    else:
        return "Not a child"

# Example usage:
image_path = "images/child_9.jpg"
prediction = predict_child_or_not(image_path)
print("Prediction:", prediction)
