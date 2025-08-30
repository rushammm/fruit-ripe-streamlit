import streamlit as st

# Import required libraries with error handling
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    st.error("TensorFlow is not installed. Please install it with: pip install tensorflow")
    HAS_TENSORFLOW = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    st.error("PIL (Pillow) is not installed. Please install it with: pip install Pillow")
    HAS_PIL = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    st.error("NumPy is not installed. Please install it with: pip install numpy")
    HAS_NUMPY = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    st.warning("OpenCV is not installed. Using PIL for image processing. Install with: pip install opencv-python")
    HAS_CV2 = False

# Set page config
st.set_page_config(
    page_title="Fruit Classification App",
    page_icon="🍎",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained fruit classification model"""
    try:
        # Try loading the .keras file first, then fall back to .h5
        model = tf.keras.models.load_model('best_fruit_model.keras')
        return model
    except:
        try:
            model = tf.keras.models.load_model('best_fruit_model.h5')
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

def preprocess_image(image, target_size=(128, 128)):
    """Preprocess the uploaded image for model prediction"""
    if not HAS_NUMPY:
        st.error("NumPy is required for image preprocessing")
        return None
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Resize image
    if HAS_CV2:
        img_resized = cv2.resize(img_array, target_size)
    else:
        # Use PIL for resizing if OpenCV is not available
        img_pil = image.resize(target_size)
        img_resized = np.array(img_pil)
    
    # Normalize pixel values to [0, 1]
    img_normalized = img_resized.astype('float32') / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def predict_fruit(model, image):
    """Make prediction on the preprocessed image"""
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        
        # Get the predicted class index
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        return predicted_class_idx, confidence, predictions[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None

# Define fruit classes (you may need to adjust these based on your model)
FRUIT_CLASSES = [
    'Overripe', 'Ripe', 'Unripe'
]

def main():
    st.title("Ripeness Detection of Fruits")
    st.write("Upload an image of a fruit to classify it using our deep learning model!")
    
    # Check if all required dependencies are available
    if not all([HAS_TENSORFLOW, HAS_PIL, HAS_NUMPY]):
        st.error("Missing required dependencies. Please install the required packages:")
        st.code("pip install streamlit tensorflow Pillow numpy opencv-python")
        st.stop()
    
    # Load the model
    model = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please check if the model files exist.")
        return
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload an image of a fruit to classify"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Add a predict button
            if st.button(" Classify Fruit", type="primary"):
                with st.spinner("Classifying..."):
                    # Make prediction
                    predicted_idx, confidence, all_predictions = predict_fruit(model, image)
                    
                    if predicted_idx is not None:
                        with col2:
                            st.header("Prediction Results")
                            
                            # Display main prediction
                            if predicted_idx < len(FRUIT_CLASSES):
                                predicted_fruit = FRUIT_CLASSES[predicted_idx]
                            else:
                                predicted_fruit = f"Class {predicted_idx}"
                            
                            st.success(f"**Predicted Fruit:** {predicted_fruit}")
                            st.info(f"**Confidence:** {confidence:.2%}")
                            
                            # Create a progress bar for confidence
                            st.progress(confidence)
                            
                            # Display top 3 predictions
                            st.subheader("Top 3 Predictions:")
                            top_indices = np.argsort(all_predictions)[::-1][:3]
                            
                            for i, idx in enumerate(top_indices):
                                if idx < len(FRUIT_CLASSES):
                                    fruit_name = FRUIT_CLASSES[idx]
                                else:
                                    fruit_name = f"Class {idx}"
                                
                                confidence_score = all_predictions[idx]
                                st.write(f"{i+1}. **{fruit_name}**: {confidence_score:.2%}")
    
    # Add information about the model
    with st.expander(" About the Model"):
        st.write("""
        This fruit classification model is a deep learning neural network trained to classify whether a fruit is ripe for not.
        
        **How to use:**
        1. Upload an image of a fruit using the file uploader
        2. Click the "Classify Fruit" button
        3. View the prediction results and confidence scores
        
        **Supported image formats:** JPG, JPEG, PNG, BMP, TIFF
        
        **Tips for better results:**
        - Use clear, well-lit images
        - Ensure the fruit is the main subject in the image
        - Avoid heavily processed or filtered images
        """)
    
    # Add footer
    st.markdown("---")

if __name__ == "__main__":
    main()
