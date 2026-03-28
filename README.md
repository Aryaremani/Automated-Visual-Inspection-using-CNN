# Automated Visual Inspection using CNN

A production-ready Streamlit web application for automated bottle inspection using deep learning. The app classifies bottle images as "Good" or "Defective" using a ResNet18 CNN model with Grad-CAM visualization.

## Features

- Binary classification (Good/Defective)
- Confidence score display
- Grad-CAM visualization for model interpretability
- Support for JPG, JPEG, and PNG images
- Dark theme with animations
- Fast inference with caching

## Live Demo

🌐 **Deployed on Streamlit Cloud:** [View App](https://automated-visual-inspection-using-cnn.streamlit.app)

## Project Structure

```
.
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── packages.txt        # System dependencies
├── model.pth          # Trained ResNet18 model
└── README.md          # This file
```

## Installation

1. Clone this repository

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Ensure your trained model file `model.pth` is in the same directory as `app.py`

2. Run the Streamlit app:
```bash
python -m streamlit run app.py
```

3. Open your browser and navigate to the URL shown in the terminal

4. Upload a bottle image and click "Run Analysis" to get results

## Model Details

- **Architecture:** ResNet18 (PyTorch)
- **Input Size:** 224x224 pixels
- **Classes:** 
  - 0 = Defective
  - 1 = Good
- **Output:** Binary classification with confidence scores

## How It Works

1. **Image Upload:** User uploads a bottle image (JPG/JPEG/PNG)
2. **Preprocessing:** Image is resized to 224x224 and converted to tensor
3. **Prediction:** ResNet18 model classifies the image
4. **Grad-CAM:** Generates visualization showing important regions
5. **Results Display:** Shows prediction, confidence, and visualizations

## Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the regions of the image that influenced the model's decision.

- **Red/Yellow areas:** High importance for the prediction
- **Blue/Purple areas:** Low importance for the prediction

## Requirements

- Python 3.8+
- PyTorch
- Streamlit
- See `requirements.txt` for complete list

## Deployment

This app is deployed on Streamlit Cloud with HTTPS and a custom domain.

## License

This project is provided as-is for educational and research purposes.
