# ELE489_ML

# Wine Classification with K-NN Algorithm

This repository demonstrates the use of the K-Nearest Neighbors (K-NN) algorithm to classify wine samples based on various features. The **Euclidean** and **Manhattan** distance metrics are used to evaluate the model's performance for different values of K.

## Steps :

1. **Data Loading**:
   - The dataset contains features like **Alcohol**, **Flavanoids**, **Proline**, etc., and their corresponding class labels.

2. **Data Visualization**:
   - Visualization of the distribution of some selected features vs. class using histograms to see if they overlap.

3. **Data Normalization**:
   - The dataset is normalized to according to min-max values of each feature.

4. **K-NN Algorithm Implementation**:
   - Implementation the K-NN algorithm from scratch, including the steps of calculating distance, finding the nearest neighbors, and performing  voting for classification.

5. **Performance of model**:
   - Evaluate the model using accuracy, confusion matrix, and classification report.

## Requirements:

- Python 3.x
- pandas
- numpy
- seaborn
- scikit-learn
- matplotlib
- Dataset from https://archive.ics.uci.edu/dataset/109/wine
