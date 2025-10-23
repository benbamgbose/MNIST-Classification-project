# MNIST Handwritten Digit Classification

This project is a foundational machine learning exercise to build and evaluate models for classifying handwritten digits from the famous MNIST dataset.

## 1. Project Overview

The goal is to accurately identify digits (0-9) from 28x28 grayscale images. This script provides an end-to-end workflow, including:
* Data Loading and Exploration
* Data Preprocessing (Scaling)
* Training and comparing three different classification models.
* In-depth model evaluation using accuracy, classification reports, and confusion matrices.
* Error analysis to understand model weaknesses.

## 2. Dataset

We use the **MNIST (Modified National Institute of Standards and Technology)** dataset, a "Hello, World!" for image classification.
* **Contents:** 70,000 images (60,000 for training, 10,000 for testing).
* **Format:** 28x28 pixel grayscale images.

*(Image: mnist_samples.png)*

## 3. Tools and Libraries

* Python 3.x
* Scikit-learn
* NumPy
* Matplotlib
* Seaborn

## 4. Project Workflow

The project is broken down into four main phases:

1.  **Data Handling and Exploration:** The dataset is loaded, its structure is examined, and sample digits are plotted to visually inspect the data.
2.  **Preprocessing and Preparation:** The data is split into standard training (60k) and testing (10k) sets. All 784 pixel features are then scaled using `StandardScaler` to normalize the data and improve model performance.
3.  **Model Building and Evaluation:** Three distinct models are trained and evaluated:
    * Logistic Regression (as a baseline)
    * K-Nearest Neighbors (KNN)
    * Random Forest Classifier
4.  **Error Analysis:** The best-performing model's misclassifications are analyzed to identify which digits are most commonly confused.

## 5. Model Performance & Results

The models were evaluated on the 10,000-image test set. The Random Forest Classifier achieved the highest accuracy.

| Model | Test Set Accuracy |
| --- | --- |
| Logistic Regression | ~91.5% |
| K-Nearest Neighbors | ~96.6% |
| **Random Forest Classifier** | **~96.9%** |

### Confusion Matrix (Random Forest)

The confusion matrix for the best model shows high performance, with most errors occurring between visually similar digits (e.g., 4s vs. 9s, 3s vs. 8s).

*(Image: confusion_matrix_Random_Forest.png)*

## 6. Error Analysis

By plotting the misclassified images, we can see *why* the model made mistakes. Many of the errors are on digits that are poorly written or.

*(Image: misclassified_digits.png)*

## 7. How to Run

1.  Clone this repository:
    ```bash
    git clone [https://github.com/YourUsername/Your-Repository-Name.git](https://github.com/YourUsername/Your-Repository-Name.git)
    ```
2.  Navigate to the project directory:
    ```bash
    cd Your-Repository-Name
    ```
3.  Install the required libraries (you can create a `requirements.txt` file with the libraries listed in section 3):
    ```bash
    pip install -r requirements.txt
    ```
4.  Run the script:
    ```bash
    python mnist_classification.py
    ```
5.  All outputs (plots like `mnist_samples.png`, `confusion_matrix...png`, etc.) will be saved to your local directory.
