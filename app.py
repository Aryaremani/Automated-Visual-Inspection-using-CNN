"""
Automated Visual Inspection using CNN - Streamlit Web App
A deep learning application for classifying bottle images as Good or Defective
using ResNet18 with Grad-CAM visualization.
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io

# Page configuration
st.set_page_config(
    page_title="Automated Visual Inspection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme with animations
def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Animated dark background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Main content container */
    .main .block-container {
        background: rgba(26, 26, 46, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 3rem 4rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        max-width: 1400px;
        margin: 2rem auto;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    [data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
    }
    
    /* Text colors */
    p, div, span, label {
        color: #d1d5db !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.875rem 2.5rem;
        font-size: 1rem;
        font-weight: 600;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 30px rgba(102, 126, 234, 0.6);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 2rem;
        border: 2px dashed rgba(102, 126, 234, 0.5);
    }
    
    /* Images */
    [data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# Class names mapping
CLASS_NAMES = {0: "Defective", 1: "Good"}

@st.cache_resource
def load_model(model_path="model.pth"):
    """
    Load the trained ResNet18 model from disk.
    Uses caching to avoid reloading on every interaction.
    
    Args:
        model_path: Path to the saved model weights
        
    Returns:
        model: Loaded PyTorch model in evaluation mode
    """
    try:
        # Initialize ResNet18 architecture
        model = models.resnet18(weights=None)
        
        # Replace final fully connected layer for binary classification
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
        # Set to evaluation mode
        model.eval()
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """
    Preprocess the uploaded image for model input.
    
    Args:
        image: PIL Image object
        
    Returns:
        tensor: Preprocessed image tensor
        original_image: Original PIL image for visualization
    """
    # Define preprocessing transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Keep original for display
    original_image = image.copy()
    
    # Apply transforms
    image_tensor = transform(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, original_image

def predict_image(model, image_tensor):
    """
    Make prediction on the preprocessed image.
    
    Args:
        model: Trained PyTorch model
        image_tensor: Preprocessed image tensor
        
    Returns:
        predicted_class: Predicted class index (0 or 1)
        confidence: Confidence score as percentage
        probabilities: Softmax probabilities for both classes
    """
    with torch.no_grad():
        # Forward pass
        outputs = model(image_tensor)
        
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get predicted class and confidence
        confidence, predicted_class = torch.max(probabilities, 1)
        
        predicted_class = predicted_class.item()
        confidence = confidence.item() * 100
        
    return predicted_class, confidence, probabilities[0].numpy()

def generate_gradcam(model, image_tensor, target_class):
    """
    Generate Grad-CAM heatmap for the predicted class.
    
    Args:
        model: Trained PyTorch model
        image_tensor: Preprocessed image tensor
        target_class: Target class for Grad-CAM
        
    Returns:
        heatmap: Grad-CAM heatmap as numpy array
    """
    # Enable gradient computation
    image_tensor.requires_grad = True
    
    # Hook to capture gradients and activations
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    # Register hooks on the last convolutional layer (layer4)
    target_layer = model.layer4[-1]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    
    # Forward pass
    output = model(image_tensor)
    
    # Backward pass for target class
    model.zero_grad()
    class_loss = output[0, target_class]
    class_loss.backward()
    
    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()
    
    # Get gradients and activations
    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]
    
    # Calculate weights (global average pooling of gradients)
    weights = np.mean(grads, axis=(1, 2))
    
    # Weighted combination of activation maps
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]
    
    # Apply ReLU to focus on positive influences
    cam = np.maximum(cam, 0)
    
    # Normalize to 0-1
    if cam.max() > 0:
        cam = cam / cam.max()
    
    return cam

def overlay_heatmap(heatmap, original_image, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on the original image.
    
    Args:
        heatmap: Grad-CAM heatmap
        original_image: Original PIL image
        alpha: Transparency factor for overlay
        
    Returns:
        overlayed_image: PIL Image with heatmap overlay
        heatmap_colored: Colored heatmap as PIL Image
    """
    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap, (original_image.width, original_image.height))
    
    # Convert heatmap to RGB using colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Convert original image to numpy array
    original_np = np.array(original_image.convert('RGB'))
    
    # Overlay heatmap on original image
    overlayed = cv2.addWeighted(original_np, 1 - alpha, heatmap_colored, alpha, 0)
    
    # Convert back to PIL Images
    overlayed_image = Image.fromarray(overlayed)
    heatmap_colored_pil = Image.fromarray(heatmap_colored)
    
    return overlayed_image, heatmap_colored_pil

# Main App UI
def main():
    # Load custom CSS
    load_custom_css()
    
    # Professional header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">Automated Visual Inspection using CNN</h1>
        <p class="main-subtitle">Deep Learning Powered Quality Control</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="description-box">
        <p class="description-text">
            This application uses a ResNet18 deep learning model to classify bottle images 
            as Good or Defective. Upload an image to receive instant predictions with 
            confidence scores and Grad-CAM visualizations showing which parts of the image 
            influenced the model's decision.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with project information
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 2rem 0 1.5rem 0;'>
            <h2 style='color: #ffffff; font-size: 1.5rem; margin: 0;'>Project Information</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='sidebar-section'>
            <div class='sidebar-title'>Model Details</div>
            <div class='sidebar-text'>
                <p style='margin: 0.5rem 0;'><strong>Architecture:</strong> ResNet18</p>
                <p style='margin: 0.5rem 0;'><strong>Framework:</strong> PyTorch</p>
                <p style='margin: 0.5rem 0;'><strong>Input Size:</strong> 224×224 pixels</p>
                <p style='margin: 0.5rem 0;'><strong>Classes:</strong> Binary (2)</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='sidebar-section'>
            <div class='sidebar-title'>Classification Classes</div>
            <div class='sidebar-text'>
                <p style='margin: 0.75rem 0;'><strong>Good:</strong> No defects detected</p>
                <p style='margin: 0.75rem 0;'><strong>Defective:</strong> Quality issues found</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='sidebar-section'>
            <div class='sidebar-title'>Features</div>
            <div class='sidebar-text'>
                <ul style='margin: 0; padding-left: 1.25rem; line-height: 1.8;'>
                    <li>Binary classification</li>
                    <li>Confidence scoring</li>
                    <li>Grad-CAM visualization</li>
                    <li>Real-time prediction</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='sidebar-section'>
            <div class='sidebar-title'>Supported Formats</div>
            <div class='sidebar-text' style='text-align: center;'>
                <p style='margin: 0;'>JPG • JPEG • PNG</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model()
    
    if model is None:
        st.markdown("""
        <div class="status-error">
            <p style='color: #ef4444; margin: 0; font-weight: 600;'>Model Loading Failed</p>
            <p style='color: #6b7280; margin: 0.5rem 0 0 0;'>Please ensure 'model.pth' exists in the application directory.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown("""
    <div class="status-success">
        <p style='color: #10b981; margin: 0; font-weight: 600;'>System Ready</p>
    </div>
    """, unsafe_allow_html=True)
    
    
    # Upload Image Section
    st.markdown('<div class="section-header">Image Upload</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Select image file",
        type=["jpg", "jpeg", "png"],
        help="Upload a bottle image for quality inspection",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="viz-label">Uploaded Image</div>', unsafe_allow_html=True)
            st.image(image, use_column_width=True)
        
        # Predict button
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("Run Analysis", type="primary", use_container_width=True)
        
        if predict_button:
            with st.spinner("Processing image..."):
                # Preprocess image
                image_tensor, original_image = preprocess_image(image)
                
                # Make prediction
                predicted_class, confidence, probabilities = predict_image(model, image_tensor)
                
                # Generate Grad-CAM
                heatmap = generate_gradcam(model, image_tensor, predicted_class)
                overlayed_image, heatmap_colored = overlay_heatmap(heatmap, original_image)
            
            st.markdown("""
            <div class="status-success">
                <p style='color: #10b981; margin: 0; font-weight: 600;'>Analysis Complete</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Prediction Results Section
            st.markdown('<div class="section-header">Inspection Results</div>', unsafe_allow_html=True)
            
            result_col1, result_col2 = st.columns([1, 1])
            
            with result_col1:
                # Display prediction
                predicted_label = CLASS_NAMES[predicted_class]
                
                if predicted_class == 1:  # Good
                    st.markdown(f"""
                    <div class='result-card result-card-success'>
                        <div class='result-title'>Good</div>
                        <div class='result-subtitle'>No Defects Detected</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:  # Defective
                    st.markdown(f"""
                    <div class='result-card result-card-error'>
                        <div class='result-title'>Defective</div>
                        <div class='result-subtitle'>Quality Issues Found</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class='confidence-card'>
                    <div class='confidence-label'>Confidence Score</div>
                    <div class='confidence-value'>{confidence:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with result_col2:
                # Display class probabilities
                st.markdown('<div class="viz-label">Class Probabilities</div>', unsafe_allow_html=True)
                
                for idx, prob in enumerate(probabilities):
                    class_name = CLASS_NAMES[idx]
                    
                    st.markdown(f"""
                    <div class='prob-card'>
                        <div class='prob-label'>{class_name}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(float(prob), text=f"{prob*100:.2f}%")
            
            # Grad-CAM Visualization Section
            st.markdown('<div class="viz-container">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Visual Attention Analysis</div>', unsafe_allow_html=True)
            st.markdown("""
            <p style='color: #6b7280; text-align: center; margin-bottom: 2rem;'>
                Heat map visualization showing regions of focus during classification. 
                Warmer colors indicate higher influence on the decision.
            </p>
            """, unsafe_allow_html=True)
            
            viz_col1, viz_col2, viz_col3 = st.columns(3)
            
            with viz_col1:
                st.markdown('<div class="viz-label">Original Image</div>', unsafe_allow_html=True)
                st.image(original_image, use_column_width=True)
            
            with viz_col2:
                st.markdown('<div class="viz-label">Attention Map</div>', unsafe_allow_html=True)
                st.image(heatmap_colored, use_column_width=True)
            
            with viz_col3:
                st.markdown('<div class="viz-label">Combined View</div>', unsafe_allow_html=True)
                st.image(overlayed_image, use_column_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class='upload-prompt'>
            <div class='upload-prompt-title'>No Image Selected</div>
            <p class='upload-prompt-text'>
                Please upload a bottle image to begin the inspection process
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
=======
"""
Automated Visual Inspection using CNN - Streamlit Web App
A deep learning application for classifying bottle images as Good or Defective
using ResNet18 with Grad-CAM visualization.
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io

# Page configuration
st.set_page_config(
    page_title="Automated Visual Inspection",
    page_icon="🔍",
    layout="wide"
)

# Class names mapping
CLASS_NAMES = {0: "Defective", 1: "Good"}

@st.cache_resource
def load_model(model_path="model.pth"):
    """
    Load the trained ResNet18 model from disk.
    Uses caching to avoid reloading on every interaction.
    
    Args:
        model_path: Path to the saved model weights
        
    Returns:
        model: Loaded PyTorch model in evaluation mode
    """
    try:
        # Initialize ResNet18 architecture
        model = models.resnet18(weights=None)
        
        # Replace final fully connected layer for binary classification
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
        # Set to evaluation mode
        model.eval()
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """
    Preprocess the uploaded image for model input.
    
    Args:
        image: PIL Image object
        
    Returns:
        tensor: Preprocessed image tensor
        original_image: Original PIL image for visualization
    """
    # Define preprocessing transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Keep original for display
    original_image = image.copy()
    
    # Apply transforms
    image_tensor = transform(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, original_image

def predict_image(model, image_tensor):
    """
    Make prediction on the preprocessed image.
    
    Args:
        model: Trained PyTorch model
        image_tensor: Preprocessed image tensor
        
    Returns:
        predicted_class: Predicted class index (0 or 1)
        confidence: Confidence score as percentage
        probabilities: Softmax probabilities for both classes
    """
    with torch.no_grad():
        # Forward pass
        outputs = model(image_tensor)
        
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get predicted class and confidence
        confidence, predicted_class = torch.max(probabilities, 1)
        
        predicted_class = predicted_class.item()
        confidence = confidence.item() * 100
        
    return predicted_class, confidence, probabilities[0].numpy()

def generate_gradcam(model, image_tensor, target_class):
    """
    Generate Grad-CAM heatmap for the predicted class.
    
    Args:
        model: Trained PyTorch model
        image_tensor: Preprocessed image tensor
        target_class: Target class for Grad-CAM
        
    Returns:
        heatmap: Grad-CAM heatmap as numpy array
    """
    # Enable gradient computation
    image_tensor.requires_grad = True
    
    # Hook to capture gradients and activations
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    # Register hooks on the last convolutional layer (layer4)
    target_layer = model.layer4[-1]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    
    # Forward pass
    output = model(image_tensor)
    
    # Backward pass for target class
    model.zero_grad()
    class_loss = output[0, target_class]
    class_loss.backward()
    
    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()
    
    # Get gradients and activations
    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]
    
    # Calculate weights (global average pooling of gradients)
    weights = np.mean(grads, axis=(1, 2))
    
    # Weighted combination of activation maps
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]
    
    # Apply ReLU to focus on positive influences
    cam = np.maximum(cam, 0)
    
    # Normalize to 0-1
    if cam.max() > 0:
        cam = cam / cam.max()
    
    return cam

def overlay_heatmap(heatmap, original_image, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on the original image.
    
    Args:
        heatmap: Grad-CAM heatmap
        original_image: Original PIL image
        alpha: Transparency factor for overlay
        
    Returns:
        overlayed_image: PIL Image with heatmap overlay
        heatmap_colored: Colored heatmap as PIL Image
    """
    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap, (original_image.width, original_image.height))
    
    # Convert heatmap to RGB using colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Convert original image to numpy array
    original_np = np.array(original_image.convert('RGB'))
    
    # Overlay heatmap on original image
    overlayed = cv2.addWeighted(original_np, 1 - alpha, heatmap_colored, alpha, 0)
    
    # Convert back to PIL Images
    overlayed_image = Image.fromarray(overlayed)
    heatmap_colored_pil = Image.fromarray(heatmap_colored)
    
    return overlayed_image, heatmap_colored_pil

# Main App UI
def main():
    # Title and description
    st.title("🔍 Automated Visual Inspection using CNN")
    st.markdown("""
    This application uses a deep learning model (ResNet18) to classify bottle images 
    as **Good** or **Defective**. Upload an image to get instant predictions with 
    confidence scores and Grad-CAM visualizations showing which parts of the image 
    influenced the model's decision.
    """)
    
    # Sidebar with project information
    with st.sidebar:
        st.header("📋 Project Information")
        st.markdown("""
        **Model:** ResNet18 (PyTorch)
        
        **Classes:**
        - ✅ Good
        - ❌ Defective
        
        **Input Size:** 224x224 pixels
        
        **Features:**
        - Binary classification
        - Confidence scoring
        - Grad-CAM visualization
        - Real-time prediction
        """)
        
        st.markdown("---")
        st.markdown("**Supported Formats:** JPG, JPEG, PNG")
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model()
    
    if model is None:
        st.error("❌ Failed to load model. Please ensure 'model.pth' exists in the app directory.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    st.markdown("---")
    
    # Upload Image Section
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a bottle image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a bottle for inspection"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_column_width=True)
        
        # Predict button
        if st.button("🔮 Predict", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                # Preprocess image
                image_tensor, original_image = preprocess_image(image)
                
                # Make prediction
                predicted_class, confidence, probabilities = predict_image(model, image_tensor)
                
                # Generate Grad-CAM
                heatmap = generate_gradcam(model, image_tensor, predicted_class)
                overlayed_image, heatmap_colored = overlay_heatmap(heatmap, original_image)
            
            st.success("✅ Prediction completed!")
            
            st.markdown("---")
            
            # Prediction Results Section
            st.header("📊 Prediction Result")
            
            result_col1, result_col2 = st.columns([1, 1])
            
            with result_col1:
                # Display prediction
                predicted_label = CLASS_NAMES[predicted_class]
                
                if predicted_class == 1:  # Good
                    st.success(f"### Prediction: {predicted_label} ✅")
                else:  # Defective
                    st.error(f"### Prediction: {predicted_label} ❌")
                
                st.metric("Confidence Score", f"{confidence:.2f}%")
            
            with result_col2:
                # Display class probabilities
                st.subheader("Class Probabilities")
                for idx, prob in enumerate(probabilities):
                    class_name = CLASS_NAMES[idx]
                    st.progress(float(prob), text=f"{class_name}: {prob*100:.2f}%")
            
            st.markdown("---")
            
            # Grad-CAM Visualization Section
            st.header("🎨 Grad-CAM Visualization")
            st.markdown("""
            Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the regions 
            of the image that were most important for the model's prediction. 
            Warmer colors (red/yellow) indicate higher importance.
            """)
            
            viz_col1, viz_col2, viz_col3 = st.columns(3)
            
            with viz_col1:
                st.subheader("Original Image")
                st.image(original_image, use_column_width=True)
            
            with viz_col2:
                st.subheader("Grad-CAM Heatmap")
                st.image(heatmap_colored, use_column_width=True)
            
            with viz_col3:
                st.subheader("Overlay")
                st.image(overlayed_image, use_column_width=True)
    
    else:
        st.info("👆 Please upload an image to begin inspection")

if __name__ == "__main__":
    main()