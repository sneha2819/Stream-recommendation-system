from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import numpy as np

# Sample dataset
data = pd.read_csv("/content/MINI_PROJECT_FINAL - Sheet1 (1).csv")

# Convert data to DataFrame
df = pd.DataFrame(data)

# Apply label encoding to categorical features
label_encoders = {}
for feature in ['Fav_Sub', 'Hobbies', 'Performance', 'Preference', 'Applications', 'Predicted_stream']:
    label_encoders[feature] = LabelEncoder()
    df[feature] = label_encoders[feature].fit_transform(df[feature])

# Split data into features and labels
X = df[['Age', 'Fav_Sub', 'Hobbies', 'Performance', 'Preference', 'Applications']].values
y = df['Predicted_stream'].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply label encoding to the target variable (Predicted_stream)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Train models
models = {
   "SVM": SVC(kernel='linear', random_state=42, probability=True),
   "Naive Bayes": GaussianNB(),
   "KNN": KNeighborsClassifier(),
   "Decision Tree": DecisionTreeClassifier()
}

results = {}
accuracies = {}
confidences = {}

print("*************************************************Analysis of Trained data:*****************************************\n")
#Compute predictions against every model
for name, model in models.items():
   model.fit(X_train, y_train_encoded)
   y_pred = model.predict(X_test)
   accuracy = accuracy_score(y_test_encoded, y_pred)
   recall = recall_score(y_test_encoded, y_pred, average = 'weighted', zero_division=1)
   precision = precision_score(y_test_encoded, y_pred, average = 'weighted', zero_division=1)
   f1 = f1_score(y_test_encoded, y_pred, average = 'weighted', zero_division=1)

  # Compute confidence (probability) of predictions
   y_proba = model.predict_proba(X_test)
   confidence = [max(proba) for proba in y_proba]

    # Store accuracy and confidence for each model
   results[name] = accuracy
   accuracies[name] = accuracy
   confidences[name] = sum(confidence) / len(confidence)

#Printing evaluation measures based on trained data
   print(f"Model: {name}")
   print(f"  Accuracy: {accuracy:.2f}")
   print(f"  Confidence: {sum(confidence) / len(confidence):.2f}")
   print(f"  Precision: {precision:.2f}")
   print(f"  Recall: {recall:.2f}")
   print(f"  F1-score: {f1:.2f}\n")

# Find best model based on accuracy
best_model_accuracy = max(accuracies, key=accuracies.get)
print(f"\nBest Model based on Accuracy: {best_model_accuracy}\n")

# Find best model based on confidence
best_model_confidence = max(results, key=results.get)
print(f"Best Model based on Confidence: {best_model_confidence}\n")

# Function to classify a person based on inputs
def classify_person(age, fav_subject, hobbies, performance, preference, application):
    fav_subject_encoded = label_encoders['Fav_Sub'].transform([fav_subject])[0]
    hobbies_encoded = label_encoders['Hobbies'].transform([hobbies])[0]
    performance_encoded = label_encoders['Performance'].transform([performance])[0]
    preference_encoded = label_encoders['Preference'].transform([preference])[0]
    application_encoded = label_encoders['Applications'].transform([application])[0]
    predictions = {}
    confidences = {}
    for name, model in models.items():
        prediction = model.predict([[age, fav_subject_encoded, hobbies_encoded, performance_encoded, preference_encoded, application_encoded]])
        predictions[name] = label_encoders['Predicted_stream'].inverse_transform(prediction)[0]
        confidence = model.predict_proba([[age, fav_subject_encoded, hobbies_encoded, performance_encoded, preference_encoded, application_encoded]])
        confidences[name] = max(confidence[0])
    return predictions, confidences

#accuracy plot generation
def generate_plot_accuracy():
  plt.figure(figsize=(10, 6))
  plt.bar(accuracies.keys(), accuracies.values(), color='skyblue')
  plt.xlabel('Model')
  plt.ylabel('Accuracy')
  plt.title('Accuracy of Models')
  plt.xticks(rotation=45)
  for i, acc in enumerate(accuracies.values()):
       plt.text(i, acc, f'{acc:.2f}', ha='center', va='bottom')
  plt.tight_layout()
  plt.show()

# Example usage
print("**************************************Accepting Test Instance:*******************************************\n")
age = int(input("\nEnter age: "))
fav_subject = input("\nWhat are your favourite subjects? \n1. Math\n2. Science\n3. English\n4. Social Science: ").upper()
hobbies = input("\nWhat are your hobbies? \n1. Sports\n2. Music\n3. Dance\n4. Reading\n5. Writing\n6. Art\n7.Other: ").upper()
performance = input("\nWhich subject are you better at? \n1. Science\n2. Commerce\n3. Humanities: ").upper()
preference = input("\nWhat do you prefer to work with? \n1. Numbers\n2. Data\n3. Abstract concepts: ").upper()
application = input("\nWhich applications do you work with better? \n1. Practical \n2. Theoretical: ").upper()

# Call classify_person() and print the return values
predicted_streams, confidences= classify_person(age, fav_subject, hobbies, performance, preference, application)
predicted_df = pd.DataFrame(list(predicted_streams.items()), columns=['Model', 'Predicted Stream'])
confidence_df = pd.DataFrame(list(confidences.items()), columns=['Model', 'Confidence'])
accuracy_df = pd.DataFrame({'Model': list(accuracies.keys()), 'Accuracy': list(accuracies.values())})

print("\n***********************************************RESULTS:***************************************************** ")
print("\nPredicted Streams: \n", predicted_df)
print("*************************************************")
print("\nConfidences: \n", confidence_df)
print("*************************************************")
print("\nAccuracies: \n", accuracy_df)
print("*************************************************")

generate_plot_accuracy()
