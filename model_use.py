import joblib

# Load the trained model
model_filename = "model.joblib"

try:
    loaded_model = joblib.load(model_filename)
    print(f"Model '{model_filename}' loaded successfully!")
except FileNotFoundError:
    print(f"Error: Model file '{model_filename}' not found. Train and save the model first.")
    exit()
    
# Function to predict text
def predict_text():
    while True:
        user_input = input("\nEnter text (or type 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            print("Exiting program")
            break

        # Predict category
        prediction = loaded_model.predict([user_input])[0]

        # Assign label
        category = "Scam" if prediction == 1 else "Non-Scam"
        
        print(f"Predicted Category: {category}")

predict_text()