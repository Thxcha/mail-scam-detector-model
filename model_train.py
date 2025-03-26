import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from pipeline import TfidfVectorizer, MultinomialNB, Pipeline

# Load dataset from CSV file
csv_filename = "emails.csv"  
try:
    df = pd.read_csv(csv_filename)
    print(f"Successfully loaded '{csv_filename}'")
except FileNotFoundError:
    print(f"Error: File '{csv_filename}' not found. Please provide a valid CSV file.")
    exit()

# Check if required columns exist
if "text" not in df.columns or "label" not in df.columns:
    print("Error: CSV file must contain 'text' and 'label' columns.")
    exit()

# Remove rows where 'text' is NaN
df = df.dropna(subset=['text'])

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Create the pipeline with custom vectorizer and classifier
vectorizer = TfidfVectorizer()
classifier = MultinomialNB()
model = Pipeline(vectorizer, classifier)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# Save the trained model pipeline
model_filename = "model.joblib"
joblib.dump(model, model_filename)
print(f"Model has been saved as '{model_filename}'")