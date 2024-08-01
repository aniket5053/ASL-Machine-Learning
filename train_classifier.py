import pickle  # Module for serializing and deserializing Python objects

# Importing necessary modules from scikit-learn for model training and evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np  # Library for numerical operations

# Load the data and labels from the pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Print the loaded data dictionary
print(data_dict)

# Convert the data and labels to NumPy arrays
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data into training and testing sets
# 75% of the data will be used for training and 25% for testing
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, shuffle=True, stratify=labels)

# Initialize the RandomForestClassifier model
model = RandomForestClassifier()

# Train the model using the training data
model.fit(x_train, y_train)

# Predict the labels for the test data
y_predict = model.predict(x_test)

# Calculate the accuracy of the predictions
score = accuracy_score(y_predict, y_test)

# Print the accuracy as a percentage
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model to a pickle file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
