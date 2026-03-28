# Automated Visual Inspection using CNN

A production-ready Streamlit web application for automated bottle inspection using deep learning. The app classifies bottle images as "Good" or "Defective" using a ResNet18 CNN model with Grad-CAM visualization.

## Features

- 🔍 Binary classification (Good/Defective)
- 📊 Confidence score display
- 🎨 Grad-CAM visualization for model interpretability
- 🖼️ Support for JPG, JPEG, and PNG images
- 💻 Clean and professional UI
- ⚡ Fast inference with caching

## Project Structure

```
.
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── model.pth          # Trained ResNet18 model (you need to provide this)
└── README.md          # This file
```

## Installation

1. Clone or download this project

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Ensure your trained model file `model.pth` is in the same directory as `app.py`

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser and navigate to the URL shown in the terminal (typically `http://localhost:8501`)

4. Upload a bottle image and click "Predict" to get results

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

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the regions of the image that influenced the model's decision. This helps understand what the model is "looking at" when making predictions.

- **Red/Yellow areas:** High importance for the prediction
- **Blue/Purple areas:** Low importance for the prediction

## Requirements

- Python 3.8+
- PyTorch 2.1.0+
- Streamlit 1.31.0+
- See `requirements.txt` for complete list

## Troubleshooting

**Model not loading:**
- Ensure `model.pth` exists in the app directory
- Verify the model was trained with ResNet18 architecture
- Check that the model has 2 output classes

**Image upload issues:**
- Ensure image format is JPG, JPEG, or PNG
- Try with a different image if one fails

**Performance issues:**
- The first prediction may be slower due to model loading
- Subsequent predictions use cached model for faster inference

## Academic Use

This application is suitable for academic submissions and demonstrations. It includes:
- Clean, well-commented code
- Professional UI design
- Model interpretability (Grad-CAM)
- Error handling
- Best practices implementation

## License

This project is provided as-is for educational and research purposes.
