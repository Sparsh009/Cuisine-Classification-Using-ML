# Cuisine Classification Using ML

## Project Overview

This repository contains two machine learning pipelines designed to classify images of dishes using the MLEnd Yummy dataset. Both projects focus on binary classification tasks—*American vs. Italian* cuisine and *Rice vs. Chips* dishes—leveraging various machine learning models to predict the cuisine based on dish images.

## Project 1: American vs. Italian Cuisine Classification

### Problem Statement
The task is to classify images of dishes as either *American* or *Italian* cuisine using image features and other dish-related attributes.

### Dataset
- **Source**: MLEnd Yummy Dataset
- **Classes**: American, Italian
- **Attributes**: Includes dish name, ingredients, cuisine, healthiness ratings, likeness, and image filenames.

### Pipeline Stages
1. **Data Collection**: Download and load the MLEnd Yummy dataset.
2. **Data Preprocessing**: Filter and encode *American* and *Italian* cuisines for classification (Italian = 0, American = 1).
3. **Feature Extraction**: Extract features such as yellow color components and GMLC (texture) features from images.
4. **Model Training**: Train models including LinearSVC, Random Forest, Logistic Regression, and k-Nearest Neighbors (k-NN).
5. **Evaluation**: Evaluate models using accuracy and confusion matrix.

### Results
- **LinearSVC**: Training Accuracy - 60.7%, Test Accuracy - 55.4%
- **Random Forest**: Training Accuracy - 66.3%, Test Accuracy - 49%
- **Logistic Regression**: Training Accuracy - 61.3%, Test Accuracy - 55%
- **k-Nearest Neighbors (k-NN)**: Training Accuracy - 76%, Test Accuracy - 47%

### Conclusion
LinearSVC and Logistic Regression performed best in the *American vs. Italian* classification task. Future work could involve using deep learning models to improve accuracy further.

---

## Project 2: Rice vs. Chips Classification

### Problem Statement
The goal is to classify images of dishes containing *Rice* or *Chips* based on the visual content of the dish, along with other attributes such as ingredients.

### Dataset
- **Source**: MLEnd Yummy Dataset
- **Classes**: Rice, Chips
- **Attributes**: Includes dish name, ingredients, cuisine, healthiness ratings, likeness, and image filenames.

### Pipeline Stages
1. **Data Collection**: Download and load the MLEnd Yummy dataset.
2. **Data Preprocessing**: Filter dishes to identify those containing *Rice* or *Chips*, downsample to achieve balance (Rice = 0, Chips = 1).
3. **Feature Extraction**: Extract features such as yellow color components and GMLC (texture) features from images.
4. **Model Training**: Train models including LinearSVC, Random Forest, Logistic Regression, and k-Nearest Neighbors (k-NN).
5. **Evaluation**: Evaluate models using accuracy and confusion matrix.

### Results
- **LinearSVC**: Training Accuracy - 63.4%, Test Accuracy - 54.2%
- **Random Forest**: Training Accuracy - 66.7%, Test Accuracy - 51%
- **Logistic Regression**: Training Accuracy - 63.4%, Test Accuracy - 54.2%
- **k-Nearest Neighbors (k-NN)**: Training Accuracy - 72.7%, Test Accuracy - 50%

### Conclusion
LinearSVC and Logistic Regression models performed best for the *Rice vs. Chips* classification task. Future improvements could include using CNNs for better image classification accuracy.

---

## How to Run

1. Clone the repository:
    ```bash
    git clone <repository_link>
    cd cuisine_classification_using_ml
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter notebooks for either project:
    ```bash
    jupyter notebook american_italian_classification.ipynb
    jupyter notebook rice_chips_classification.ipynb
    ```

## Conclusion

Both projects aim to classify dish images into binary categories using machine learning techniques. The models achieved moderate performance, with potential for improvement using advanced image processing techniques or deep learning models such as Convolutional Neural Networks (CNNs).
