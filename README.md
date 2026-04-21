# Automated Visual Inspection using CNN

A production-ready Streamlit web application for automated bottle inspection using deep learning. The app classifies bottle images as "Good" or "Defective" using a ResNet18 CNN model with Grad-CAM visualization.

## Features

- Binary classification (Good/Defective)
- confidence score display
- Grad-CAM visualization for model interpretability
- Support for JPG, JPEG, and PNG images
- Dark theme with animations
- Fast inference with caching

## Data Understanding and EDA 

The project uses the bottle category from the MVTec Anomaly Detection dataset. The dataset contains training images of normal bottles and test images with both normal and defective samples.

Defects include:
- broken_large
- broken_small
- contamination

Initial exploration shows that the dataset is imbalanced, with more normal images than defective ones. Sample images and class distribution were analyzed to understand the dataset before training the model.

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
## Model Development

A pretrained ResNet18 model was used for binary classification of bottle images into **good** and **defective** classes. Transfer learning was applied by replacing the final fully connected layer with a two-class output layer.  

To address class imbalance, weighted cross-entropy loss was used during training. The model was trained for multiple epochs, and performance was monitored using training loss and accuracy.  

Grad-CAM was further integrated to improve interpretability by highlighting the image regions that contributed most to the model’s prediction.

## Evaluation Summary

The model achieved a test accuracy of approximately 78.8%. The confusion matrix shows that all defective samples were correctly identified, achieving a recall of 1.00 for the defective class.

However, some normal images were misclassified as defective, indicating that the model is conservative and prioritizes defect detection. This behavior is acceptable in industrial scenarios where detecting defects is more critical than avoiding false alarms.

### Training Configuration

The ResNet18 model was trained using transfer learning with weighted cross-entropy loss to handle class imbalance. The optimizer used was Adam, and performance was monitored across multiple epochs using loss and training accuracy.
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
https://automated-visual-inspection-using-cnn.streamlit.app/

<img width="1701" height="763" alt="Screenshot 2026-04-20 222407" src="https://github.com/user-attachments/assets/0ecead95-31fa-481b-afcb-668b92407deb" />


## License

This project is provided as-is for educational and research purposes.
