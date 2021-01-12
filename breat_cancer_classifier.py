from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#load data
breast_cancer_data = load_breast_cancer()

#preview data
#print(breast_cancer_data.data[0])
#print(breast_cancer_data.feature_names)
#print(breast_cancer_data.target)
#print(breast_cancer_data.target_names)

#split the data between training and validation
training_data, validation_data, training_labels, validation_labels = train_test_split(
  breast_cancer_data.data,
  breast_cancer_data.target,
  test_size=0.2,
  random_state=80
)

#show the training data and its labels
#print(len(training_data))
#print(len(training_labels))

#initialize variables to be used for finding the most accurate number of neighbors for the K Nearest Neighbors classifier model
k_list = range(1,101)
accuracies = []

#create the KNN Classifier model and fit it to the training data, then use it on the validation data and see how accurate it is
for k in range(1,101):
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_data, training_labels)
  accuracies.append(classifier.score(validation_data, validation_labels))
  #print(accuracies)

#plot the accuracies of the model depending on the number of neighbors used for classification
plt.plot(k_list, accuracies)
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.title('Breast Cancer Classifier Accuracy')
plt.show()

#Based on the plot, choose the optimum number of neighbors and create a new classification model using that number. 
classifier2 = KNeighborsClassifier(n_neighbors = 8)
classifier2.fit(training_data, training_labels)
print(classifier2.score(validation_data, validation_labels))