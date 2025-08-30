import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image
import json

st.set_page_config(page_title="Fruit Ripeness Classifier", layout="wide")


@st.cache_data
def load_class_names():
    with open("class_names.json", "r") as f:
        return json.load(f)


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = load_class_names()
    num_classes = len(class_names)

    model = models.resnet50(pretrained=False)
    model.fc = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes)
    )

    model.load_state_dict(
        torch.load("best_resnet_model_cleaned.pth", map_location=device)
    )
    model = model.to(device)
    model.eval()

    return model, device


def preprocess_image(image, device):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    if image.mode != "RGB":
        image = image.convert("RGB")

    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor


def predict(model, image_tensor, class_names, device):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_idx].item()

        probs_np = probabilities[0].cpu().numpy()

    st.write(f"**Debug Info:**")
    st.write(f"Device: {device}")
    st.write(f"Input shape: {image_tensor.shape}")
    st.write(
        f"Raw outputs range: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]"
    )
    st.write(f"Top 3 predictions:")
    top_3_idx = torch.topk(probabilities[0], 3).indices.cpu().numpy()
    for idx in top_3_idx:
        st.write(f"  {class_names[idx]}: {probs_np[idx]:.3f}")
    st.write("---")

    return class_names[predicted_idx], confidence, probs_np


def main():
    st.title("Fruit Ripeness Classifier")
    st.write("Upload an image to classify fruit ripeness (PyTorch ResNet50)")

    class_names = load_class_names()

    with st.spinner("Loading PyTorch model..."):
        try:
            model, device = load_model()
            st.success(f"PyTorch ResNet50 model loaded successfully on {device}")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.info(
                "Make sure 'best_resnet_model_cleaned.pth' exists in the current directory"
            )
            return

    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width="stretch")
            st.write(f"**Image Info:**")
            st.write(f"Size: {image.size[0]} x {image.size[1]} pixels")
            st.write(f"Mode: {image.mode}")

        with col2:
            with st.spinner("Processing image..."):
                try:
                    image_tensor = preprocess_image(image, device)

                    predicted_class, confidence, probabilities = predict(
                        model, image_tensor, class_names, device
                    )

                    st.write(f"**Prediction:** {predicted_class}")
                    st.write(f"**Confidence:** {confidence:.2%}")

                    st.write("**All Class Probabilities:**")
                    for i, (label, prob) in enumerate(zip(class_names, probabilities)):
                        if i == np.argmax(probabilities):
                            st.write(f"**{label}: {prob:.2%}**")
                        else:
                            st.write(f"{label}: {prob:.2%}")

                    st.write("**Interpretation:**")
                    if confidence > 0.8:
                        st.success(f"High confidence prediction: {predicted_class}")
                    elif confidence > 0.6:
                        st.info(f"Moderate confidence prediction: {predicted_class}")
                    else:
                        st.warning(f"Low confidence prediction: {predicted_class}")

                except Exception as e:
                    st.error(f"Error during prediction: {e}")


if __name__ == "__main__":
    main()
