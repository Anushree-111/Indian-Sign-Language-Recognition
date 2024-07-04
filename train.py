import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from pickle file
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Ensure all elements are lists of 84 floats
def validate_and_flatten(element):
    if isinstance(element, list) and len(element) == 84:
        return [float(x) for x in element]
    else:
        raise ValueError(f"Invalid element structure: {element}")

# Apply validation and flattening
validated_data = []
invalid_elements = 0
for i, el in enumerate(data_dict['data']):
    try:
        validated_data.append(validate_and_flatten(el))
    except ValueError as e:
        invalid_elements += 1
        print(f"Skipping invalid element at index {i}: {e}")

print(f"Number of invalid elements skipped: {invalid_elements}")

# Convert validated data and labels to numpy arrays
data = np.array(validated_data, dtype=np.float32)
labels = np.array(data_dict['labels'][:len(validated_data)])  # Ensure labels array matches the length of valid data

# Check the shape of data and labels
print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions on the test set
y_predict = model.predict(x_test)

# Calculate and print the accuracy
score = accuracy_score(y_test, y_predict)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
