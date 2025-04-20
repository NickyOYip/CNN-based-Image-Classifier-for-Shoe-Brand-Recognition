# CNN-based Image Classifier for Shoe Brand Recognition

## Overview
This project implements a Convolutional Neural Network (CNN) to classify shoe images into three brands: Nike, Adidas, and Converse. The model is trained and evaluated using a custom dataset and provides training results, learning curves, and a saved best model.

## Project Report (Summary)
This project involved iterative development of a CNN-based image classifier for shoe brand recognition (Nike, Adidas, Converse) without transfer learning. The process included:
- Data augmentation to address the small dataset size.
- Multiple model architectures, progressively increasing complexity and regularization.
- Hyperparameter tuning, early stopping, and dropout to combat overfitting.
- The final model achieved up to 74% validation accuracy with 355,363 parameters.

**Conclusion:**
Iterative model refinement and regularization were key to improving performance on a small dataset. For further improvement, consider transfer learning or expanding the dataset.

## Example Learning Curve
Below is the learning curve from the final training run (folder `13`):

![Learning Curve](training%20result/13/learning_curves.png)

## Dataset
- Located in: `Nike_Adidas_converse_Shoes_image_dataset/`
- Structure: 
  - `train/` — Training images, organized by brand.
  - `validate/` — Validation images, organized by brand.

## Training Results
- Each run (folders `0` to `13` in `training result/`) contains:
  - `learning_curves.png` — Training/validation loss and accuracy plots.
  - `model_results_dynamic.xlsx` — Hyperparameters and training history.

## Model
- Architecture: Sequential CNN with Conv2D, MaxPooling2D, Dropout, and Dense layers.
- Best model saved as: `13314481_best_model_.h5`

## How to Run
1. Open `Machine_learning_project_Jnotebook.ipynb` in Jupyter or Google Colab.
2. Mount your Google Drive if using Colab.
3. Update dataset paths if needed.
4. Run all cells to train, evaluate, and save the model.

## Requirements
- Python 3.x
- TensorFlow
- Keras
- Matplotlib
- Pandas
- openpyxl

## Results
- Training and validation accuracy/loss are visualized in each run’s `learning_curves.png`.
- Hyperparameters and training history are saved in `model_results_dynamic.xlsx`.
- The best model can be loaded from `13314481_best_model_.h5` for inference.
