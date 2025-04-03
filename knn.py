import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn import metrics as mt
from sklearn.metrics import classification_report
column_names = [
    "Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
    "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins",
    "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"
]


file_path = "wine/wine.data"
wine_df = pd.read_csv(file_path, header=None, names=column_names)
# Visualize some features to see class separability

features_to_plot = ["Alcohol", "Flavanoids", "Proline"]

for feature in features_to_plot:
    plt.figure(figsize=(7, 4))
    for class_label in sorted(wine_df["Class"].unique()):
        sns.histplot(wine_df[wine_df["Class"] == class_label][feature],
                     label=f"Class {class_label}", kde=True, bins=15, alpha=0.5)
    plt.title(f"{feature} Distribution by Class")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.legend(title="Class")
    plt.show()


print(wine_df.head())

def normalize_min_max(_df):
    _df = _df.copy()
    for feature in _df.columns[1:]:
        min_val = _df[feature].min()
        max_val = _df[feature].max()
        _df[feature] = (_df[feature] - min_val) / (max_val - min_val)
    return _df

normalized_wine_df = normalize_min_max(wine_df)
#Normalized class and feature dataframes
x = normalized_wine_df.iloc[:, 1:]
y = normalized_wine_df.iloc[:, 0]

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=0,stratify=y)

#Convert the dataframes to arrays to perform operations easier
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)


def calculate_distance(train, test_point, metric):
    distances = []
    for row in train:
        if metric.lower() == "euclidean":
            dist = np.sqrt(np.sum((row - test_point) ** 2))
        elif metric.lower() == "manhattan":
            dist = np.sum(np.abs(row - test_point))
        distances.append(dist)

    return pd.DataFrame({'dist': distances})

def nearest_neighbors(distance_point, K):
    df_nearest= distance_point.sort_values(by=['dist'], axis=0)
    ## Take only the first K neighbors
    df_nearest= df_nearest[:K]
    return df_nearest

def voting(df_nearest, y_train):
    ## Use the Counter Object to get the labels with K nearest neighbors.
    counter_vote= Counter(y_train[df_nearest.index])

    y_pred= counter_vote.most_common()[0][0]
    return y_pred

def knn_algorithm(x_train, y_train, x_test, K, metric):
    y_pred=[]
    # Loop over all the test set and perform the three steps
    for x_test_point in x_test:
      distance_point  = calculate_distance(x_train, x_test_point, metric)
      df_nearest_point= nearest_neighbors(distance_point, K)
      y_pred_point    = voting(df_nearest_point, y_train)
      y_pred.append(y_pred_point)

    return y_pred

k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17]
metrics = ["euclidean", "manhattan"]
accuracy_results = {}

for metric in metrics:
    accuracy_results[metric] = []
    for k in k_values:
        y_pred = knn_algorithm(x_train, y_train, x_test, k, metric)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_results[metric].append(accuracy)

# Display accuracy results
print(accuracy_results)

plt.figure(figsize=(10, 6))
for metric in metrics:
    plt.plot(k_values, accuracy_results[metric], marker='o', label=metric)

plt.title("Accuracy vs. K for Different Distance Metrics")
plt.xlabel("K (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.legend(title="Distance Metric")
plt.grid(True)
plt.show()

# Compute confusion matrix
chosen_k = 11
chosen_metric = "euclidean"
y_pred = knn_algorithm(x_train, y_train, x_test,chosen_k, chosen_metric)
# Display confusion matrix
confusion_matrix = mt.confusion_matrix(y_test, y_pred)
cm_display = mt.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=np.unique(y_test))
cm_display.plot(cmap='Blues')
plt.title(f"Confusion Matrix (k={chosen_k}, {chosen_metric} distance)")
plt.show()
#Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

