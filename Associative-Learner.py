import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings

# Ignore some future warnings to keep the output clean
warnings.filterwarnings("ignore", category=FutureWarning)


def train_and_evaluate_associative_learner():
    """
    A complete function to load data, train the model, test, and analyze the results.
    """
    try:
        # --- Step 1: Loading agent_experience_log.csv dataset... ---
        print("--- Step 1: Loading agent_experience_log.csv dataset... ---")
        data = pd.read_csv('agent_experience_log.csv')
        print(f"Successfully loaded {len(data)} decision records.")

        # --- Step 2: Preparing training and testing data... ---
        print("\n--- Step 2: Preparing training and testing data... ---")

        # Define the model's inputs (features) and learning target (label)
        features = ['is_in_A_zone', 'is_in_B_zone', 'is_occluded']
        target = 'correct_action_label'

        # Strictly divide the data into A-Trial and B-Trial phases based on simulation steps
        A_trial_data = data[data['sim_step'] < 70000]
        B_trial_data = data[data['sim_step'] >= 70000]

        if A_trial_data.empty:
            print("Error: No data found for the A-Trial phase (simulation steps < 70000). Please check the data generation process.")
            return
        if B_trial_data.empty:
            print("Error: No data found for the B-Trial phase (simulation steps >= 70000). Please check the data generation process.")
            return

        # Prepare the training set (A-Trials)
        X_train = A_trial_data[features]
        y_train = A_trial_data[target]

        # Prepare the test set (B-Trials)
        X_test = B_trial_data[features]
        y_test = B_trial_data[target]

        print(f"Training set size (A-Trials): {len(X_train)} records")
        print(f"Test set size (B-Trials): {len(X_test)} records")

        # --- Step 3: Training the Associative Learner (neural network model)... ---
        print("\n--- Step 3: Training the Associative Learner (neural network model)... ---")

        # Initialize a simple MLP Classifier. This is our "Associative Learner".
        associative_learner = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500, random_state=1, activation='relu')

        # Train the model using the A-Trial data
        associative_learner.fit(X_train, y_train)
        print("Model training complete.")

        # --- Step 4: Evaluating the model on B-Trial test data... ---
        print("\n--- Step 4: Evaluating the model on B-Trial test data... ---")

        # Make predictions on the B-Trial data
        predictions = associative_learner.predict(X_test)

        print("\nOverall Classification Report on B-Trials:")
        print("(Note: '0' represents Action A, '1' represents Action B)")
        print(classification_report(y_test, predictions, zero_division=0))

        print("\nConfusion Matrix for B-Trials:")
        print("(Rows: True Labels, Columns: Predicted Labels)")
        print(confusion_matrix(y_test, predictions))

        # --- Step 5: In-depth analysis of the critical first B-Trial... ---
        print("\n--- Step 5: In-depth analysis of the critical first B-Trial... ---")

        # Get the input state for the very first B-Trial
        first_b_trial_input = X_test.iloc[0:1]
        first_b_trial_prediction = associative_learner.predict(first_b_trial_input)
        first_b_trial_probs = associative_learner.predict_proba(first_b_trial_input)

        print(f"\nInput state for the critical test (first B-Trial):")
        print(first_b_trial_input)

        print(f"\nModel's predicted action for this state: {first_b_trial_prediction[0]} (0=Reach for A, 1=Reach for B)")
        print(f"The correct action in this context is: {y_test.iloc[0]}")
        print(f"Model's confidence probabilities for [Action A, Action B] are: {first_b_trial_probs[0]}")

        # --- Final Conclusion ---
        print("\n==================== EXPERIMENTAL CONCLUSION ====================")
        if first_b_trial_prediction[0] == 0 and y_test.iloc[0] == 1:
            print("Success! The model exhibits a classic A-non-B perseverative error.")
            print("Analysis: Despite the input state clearly indicating Zone B, the model developed an overwhelmingly strong statistical association during the A-trials,")
            print(f"and therefore, with {first_b_trial_probs[0][0]:.2%} high confidence, it incorrectly persisted in choosing Action 'A'.")
            print("This provides strong evidence that the mechanism of error for this model is the strength of its statistical associations.")
        elif first_b_trial_prediction[0] == y_test.iloc[0]:
            print("The model correctly predicted Action 'B'.")
            print("Analysis: This may indicate that the model's complexity or the training data was not sufficient to induce a perseverative error.")
            print("You could try simplifying the model (e.g., reducing hidden_layer_sizes) or review the data generation process.")
        else:
            print("The model's prediction was unexpected. Please review the analysis steps above.")
        print("==================================================")

    except FileNotFoundError:
        print("\nError: Could not find 'agent_experience_log.csv'.")
        print("Please ensure you have successfully run main.py and that the CSV file is in the same directory as this script.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during execution: {e}")


# Execute the main function when the script is run
if __name__ == "__main__":
    train_and_evaluate_associative_learner()