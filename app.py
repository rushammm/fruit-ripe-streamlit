import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Configure Streamlit page
st.set_page_config(
    page_title="Fruit Ripeness Classifier",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define class labels (based on model output with 3 classes)
CLASS_LABELS = ["Ripe", "Unripe", "Overripe"]

@st.cache_resource
def load_model(model_path):
    """Load and cache the TensorFlow model"""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image, target_size=(128, 128)):
    """Preprocess image for model prediction"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image to model input size
        image = image.resize(target_size)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Normalize pixel values to [0, 1]
        image_array = image_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def predict_ripeness(model, image_array):
    """Make prediction using the loaded model"""
    try:
        # Make prediction
        predictions = model.predict(image_array, verbose=0)
        
        # Get class probabilities
        probabilities = tf.nn.softmax(predictions[0]).numpy()
        
        # Get predicted class
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = CLASS_LABELS[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]
        
        return predicted_class, confidence, probabilities
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None, None

def main():
    st.title("Fruit Ripeness Classifier")
    st.markdown("""
    Upload an image of a fruit and get an AI-powered analysis of its ripeness!
    
    **Model Information:**
    - Architecture: MobileNetV2 (optimized for mobile/edge devices)
    - Input Size: 128x128 pixels
    - Classes: Ripe, Unripe, Overripe
    """)
    
    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox(
        "Choose model format:",
        ["best_fruit_model.keras", "best_fruit_model.h5"],
        help="Both models are identical, choose your preferred format"
    )
    
    # Load the selected model
    with st.spinner(f"Loading {model_choice}..."):
        model = load_model(model_choice)
    
    if model is None:
        st.error("Failed to load model. Please check if the model files exist.")
        return
    
    st.success(f"✅ Model loaded successfully: {model_choice}")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a fruit image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image in PNG, JPG, or JPEG format"
    )
    
    if uploaded_file is not None:
        # Create two columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            
            # Load and display the image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_container_width=True)
            
            # Show image details
            st.info(f"""
            **Image Details:**
            - Size: {image.size[0]} x {image.size[1]} pixels
            - Mode: {image.mode}
            - Format: {image.format}
            """)
        
        with col2:
            st.subheader("AI Analysis")
            
            # Preprocess the image
            with st.spinner("Preprocessing image..."):
                processed_image = preprocess_image(image)
            
            if processed_image is not None:
                # Make prediction
                with st.spinner("Analyzing ripeness..."):
                    predicted_class, confidence, probabilities = predict_ripeness(model, processed_image)
                
                if predicted_class is not None:
                    # Display main prediction
                    st.success(f"**Prediction: {predicted_class}**")
                    st.info(f"**Confidence: {confidence:.2%}**")
                    
                    # Display confidence meter
                    st.metric("Confidence Score", f"{confidence:.2%}")
                    
                    # Display all class probabilities
                    st.subheader("Class Probabilities")
                    
                    for i, (label, prob) in enumerate(zip(CLASS_LABELS, probabilities)):
                        # Color code based on prediction
                        if i == np.argmax(probabilities):
                            st.success(f"**{label}: {prob:.2%}**")
                        else:
                            st.write(f"{label}: {prob:.2%}")
                        
                        # Progress bar for each class
                        st.progress(float(prob))
                    
                    # Interpretation
                    st.subheader("Interpretation")
                    if predicted_class == "Ripe":
                        st.success("This fruit appears to be **ripe** and ready to eat!")
                    elif predicted_class == "Unripe":
                        st.warning("This fruit appears to be **unripe**. You might want to wait a bit longer.")
                    elif predicted_class == "Overripe":
                        st.error("This fruit appears to be **overripe**. It might be past its prime.")
                    
                    # Confidence interpretation
                    if confidence > 0.8:
                        st.info("High confidence prediction!")
                    elif confidence > 0.6:
                        st.info("Moderate confidence prediction.")
                    else:
                        st.warning("Low confidence prediction. The image might be unclear or the fruit might be borderline between categories.")

 

    
if __name__ == "__main__":
    main()