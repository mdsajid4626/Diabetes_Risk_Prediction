import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier






# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "dataset/diabetes.csv"


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    """Load the Pima Indians Diabetes dataset."""

    print("=" * 60)
    print("DIABETES RISK PREDICTION")
    print("=" * 60)

    print("\nLoading dataset...")

    df = pd.read_csv(DATASET_PATH)

    print("Dataset loaded successfully.")

    return df


# ============================================================
# DATASET INSPECTION
# ============================================================

def inspect_dataset(df):
    """Inspect dataset structure and basic statistics."""

    print("\n" + "=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(list(df.columns))

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nBasic statistics:")
    print(df.describe())


# ============================================================
# CHECK ZERO VALUES
# ============================================================

def check_zero_values(df):
    """
    Check zero values in medical features.

    In this dataset, zero values in certain medical
    measurements can represent hidden missing values.
    """

    print("\n" + "=" * 60)
    print("ZERO VALUE ANALYSIS")
    print("=" * 60)

    medical_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    for column in medical_columns:
        zero_count = (df[column] == 0).sum()

        print(
            f"{column:<25} : {zero_count} zero values"
        )


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

def check_target_distribution(df):
    """Display the distribution of the Outcome target."""

    print("\n" + "=" * 60)
    print("TARGET DISTRIBUTION")
    print("=" * 60)

    print("\nOutcome counts:")

    print(df["Outcome"].value_counts())

    print("\nOutcome percentages:")

    print(
        (df["Outcome"].value_counts(normalize=True) * 100)
        .round(2)
    )

# ============================================================
# DATA PREPROCESSING
# ============================================================

def preprocess_data(df):
    """
    Handle hidden missing values represented by zero.

    For medical measurements, zero is treated as a missing value.
    Missing values are replaced using the median of each feature.
    """

    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    df = df.copy()

    # Medical features where zero is not a valid measurement
    medical_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    print("\nConverting hidden zero values to missing values...")

    for column in medical_columns:
        zero_count = (df[column] == 0).sum()

        # Replace zero with NaN
        df[column] = df[column].replace(0, pd.NA)

        print(
            f"{column:<25} : {zero_count} values converted"
        )

    print("\nImputing missing values using median...")

    for column in medical_columns:

        missing_before = df[column].isna().sum()

        if missing_before > 0:

            median_value = df[column].median()

            df[column] = df[column].fillna(median_value)

            print(
                f"{column:<25} : "
                f"{missing_before} missing → "
                f"median {median_value:.2f}"
            )

    print("\nRemaining missing values:")

    print(df.isnull().sum())

    print("\nData preprocessing completed.")

    return df
# ============================================================
# TRAIN-TEST SPLIT
# ============================================================

def split_data(df):
    """
    Separate features and target, then create a stratified
    80/20 train-test split.
    """

    print("\n" + "=" * 60)
    print("TRAIN-TEST SPLIT")
    print("=" * 60)

    # Features
    X = df.drop("Outcome", axis=1)

    # Target
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining features : {X_train.shape}")
    print(f"Testing features  : {X_test.shape}")
    print(f"Training target   : {y_train.shape}")
    print(f"Testing target    : {y_test.shape}")

    print("\nTraining target distribution:")
    print(y_train.value_counts(normalize=True).round(3))

    print("\nTesting target distribution:")
    print(y_test.value_counts(normalize=True).round(3))

    print("\nTrain-test split completed.")

    return X_train, X_test, y_train, y_test


# ============================================================
# FEATURE SCALING
# ============================================================

def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler.

    The scaler is fitted only on training data to prevent
    data leakage.
    """

    print("\n" + "=" * 60)
    print("FEATURE SCALING")
    print("=" * 60)

    scaler = StandardScaler()

    # Fit ONLY on training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform test data using the same scaler
    X_test_scaled = scaler.transform(X_test)

    print("\nStandardScaler applied successfully.")

    print(f"Scaled training data : {X_train_scaled.shape}")
    print(f"Scaled testing data  : {X_test_scaled.shape}")

    print("\nFeature scaling completed.")

    return X_train_scaled, X_test_scaled, scaler

# ============================================================
# TRAIN ML MODELS
# ============================================================

def train_models(X_train_scaled, y_train):
    """
    Train KNN, Logistic Regression, and Decision Tree models.
    """

    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        )
    }

    trained_models = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train_scaled, y_train)

        trained_models[name] = model

        print(f"{name} training completed.")

    print("\nAll models trained successfully.")

    return trained_models

# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_models(trained_models, X_test_scaled, y_test):
    """
    Evaluate all trained models using:
    Accuracy, Precision, Recall, F1-score and Confusion Matrix.
    """

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix
    )

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    results = {}

    for name, model in trained_models.items():

        # Make predictions on unseen test data
        y_pred = model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        matrix = confusion_matrix(y_test, y_pred)

        results[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": matrix
        }

        print(f"\n{name}")
        print("-" * 40)
        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1-score  : {f1:.4f}")

        print("\nConfusion Matrix:")
        print(matrix)

    return results

# ============================================================
# THRESHOLD TUNING
# ============================================================

def tune_threshold(model, X_test_scaled, y_test, model_name):
    """
    Evaluate different probability thresholds for a model.

    Lower thresholds generally increase recall while potentially
    increasing false positives.
    """

    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix
    )

    print("\n" + "=" * 60)
    print(f"THRESHOLD TUNING - {model_name}")
    print("=" * 60)

    # Get probability of the positive class (diabetes)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    thresholds = [
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60
    ]

    results = []

    print(
        f"\n{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'FP':<8}"
        f"{'FN':<8}"
    )

    print("-" * 64)

    for threshold in thresholds:

        # Convert probabilities into predictions
        y_pred = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        matrix = confusion_matrix(y_test, y_pred)

        tn, fp, fn, tp = matrix.ravel()

        print(
            f"{threshold:<12.2f}"
            f"{precision:<12.4f}"
            f"{recall:<12.4f}"
            f"{f1:<12.4f}"
            f"{fp:<8}"
            f"{fn:<8}"
        )

        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positives": fp,
            "false_negatives": fn
        })

    return results

# ============================================================
# BUSINESS CAPACITY ANALYSIS
# ============================================================

def business_capacity_analysis(trained_models, X_test_scaled, y_test):
    """
    Evaluate models under a limited testing capacity.

    The clinic can test approximately 30% of the test population.
    Patients with the highest predicted diabetes probabilities
    are prioritized for testing.
    """

    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix
    )

    print("\n" + "=" * 60)
    print("BUSINESS CAPACITY ANALYSIS")
    print("=" * 60)

    capacity = 0.30
    total_patients = len(y_test)
    testing_capacity = int(total_patients * capacity)

    print(f"\nTotal test patients       : {total_patients}")
    print(f"Testing capacity          : {capacity * 100:.0f}%")
    print(f"Maximum patients tested   : {testing_capacity}")

    business_results = {}

    for model_name, model in trained_models.items():

        probabilities = model.predict_proba(X_test_scaled)[:, 1]

        # Select patients with highest predicted risk
        ranked_indices = probabilities.argsort()[::-1]

        selected_indices = ranked_indices[:testing_capacity]

        y_pred = [0] * total_patients

        for index in selected_indices:
            y_pred[index] = 1

        y_pred = pd.Series(y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        matrix = confusion_matrix(y_test, y_pred)

        tn, fp, fn, tp = matrix.ravel()

        business_results[model_name] = {
            "capacity": testing_capacity,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "confusion_matrix": matrix
        }

        print(f"\n{model_name}")
        print("-" * 40)
        print(f"Patients selected : {testing_capacity}")
        print(f"Precision         : {precision:.4f}")
        print(f"Recall            : {recall:.4f}")
        print(f"F1-score          : {f1:.4f}")
        print(f"True Positives    : {tp}")
        print(f"False Positives   : {fp}")
        print(f"False Negatives   : {fn}")

        print("\nConfusion Matrix:")
        print(matrix)

    return business_results

# ============================================================
# MODEL EVALUATION VISUALIZATION
# ============================================================

def create_visualizations(trained_models, X_test_scaled, y_test):
    """
    Create visualizations for model comparison and
    final business-capacity evaluation.
    """

    from sklearn.metrics import (
        confusion_matrix,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score
    )

    print("\n" + "=" * 60)
    print("CREATING MODEL VISUALIZATIONS")
    print("=" * 60)

    model_names = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    # --------------------------------------------------------
    # Calculate performance metrics
    # --------------------------------------------------------

    for model_name, model in trained_models.items():

        y_pred = model.predict(X_test_scaled)

        model_names.append(model_name)

        accuracies.append(
            accuracy_score(y_test, y_pred)
        )

        precisions.append(
            precision_score(
                y_test,
                y_pred,
                zero_division=0
            )
        )

        recalls.append(
            recall_score(
                y_test,
                y_pred,
                zero_division=0
            )
        )

        f1_scores.append(
            f1_score(
                y_test,
                y_pred,
                zero_division=0
            )
        )

    # --------------------------------------------------------
    # 1. Model Performance Comparison
    # --------------------------------------------------------

    x = range(len(model_names))
    width = 0.2

    plt.figure(figsize=(10, 6))

    plt.bar(
        [i - 1.5 * width for i in x],
        accuracies,
        width,
        label="Accuracy"
    )

    plt.bar(
        [i - 0.5 * width for i in x],
        precisions,
        width,
        label="Precision"
    )

    plt.bar(
        [i + 0.5 * width for i in x],
        recalls,
        width,
        label="Recall"
    )

    plt.bar(
        [i + 1.5 * width for i in x],
        f1_scores,
        width,
        label="F1-score"
    )

    plt.xticks(
        list(x),
        model_names
    )

    plt.ylabel("Score")
    plt.xlabel("Machine Learning Model")
    plt.title("Model Performance Comparison")

    plt.ylim(0, 1)

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "model_performance_comparison.png",
        dpi=300
    )

    plt.show()

    print("Saved: model_performance_comparison.png")

    # --------------------------------------------------------
    # 2. Confusion Matrix - Final Business Model
    # --------------------------------------------------------

    final_model = trained_models["KNN"]

    # Select top 30% patients according to predicted risk
    probabilities = final_model.predict_proba(
        X_test_scaled
    )[:, 1]

    testing_capacity = int(
        len(y_test) * 0.30
    )

    ranked_indices = probabilities.argsort()[::-1]

    selected_indices = ranked_indices[
        :testing_capacity
    ]

    business_predictions = [0] * len(y_test)

    for index in selected_indices:
        business_predictions[index] = 1

    matrix = confusion_matrix(
        y_test,
        business_predictions
    )

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Not Diabetic",
            "Diabetic"
        ],
        yticklabels=[
            "Not Diabetic",
            "Diabetic"
        ]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(
        "KNN Confusion Matrix - 30% Testing Capacity"
    )

    plt.tight_layout()

    plt.savefig(
        "knn_confusion_matrix.png",
        dpi=300
    )

    plt.show()

    print("Saved: knn_confusion_matrix.png")

    # --------------------------------------------------------
    # 3. Recall Comparison
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.bar(
        model_names,
        recalls
    )

    plt.ylabel("Recall")
    plt.xlabel("Machine Learning Model")
    plt.title("Recall Comparison")

    plt.ylim(0, 1)

    plt.tight_layout()

    plt.savefig(
        "recall_comparison.png",
        dpi=300
    )

    plt.show()

    print("Saved: recall_comparison.png")

    print("\nVisualization completed successfully.")


# ============================================================
# SAVE FINAL MODEL
# ============================================================

def save_final_model(trained_models, scaler):
    """
    Save the selected KNN model and feature scaler
    for future predictions.
    """

    print("\n" + "=" * 60)
    print("SAVING FINAL MODEL")
    print("=" * 60)

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Select KNN based on business-capacity analysis
    final_model = trained_models["KNN"]

    # Save trained KNN model
    joblib.dump(
        final_model,
        "models/knn_model.pkl"
    )

    # Save feature scaler
    joblib.dump(
        scaler,
        "models/scaler.pkl"
    )

    print("\nFinal model saved:")
    print("models/knn_model.pkl")

    print("\nFeature scaler saved:")
    print("models/scaler.pkl")

    print("\nModel saving completed successfully.")

# ============================================================
# MAIN
# ============================================================

def main():

    try:

        df = load_dataset()

        inspect_dataset(df)

        check_zero_values(df)

        check_target_distribution(df)

        df = preprocess_data(df)

        X_train, X_test, y_train, y_test = split_data(df)

        X_train_scaled, X_test_scaled, scaler = scale_features(
            X_train,
            X_test
        )

        trained_models = train_models(
            X_train_scaled,
            y_train
        )

        results = evaluate_models(
            trained_models,
            X_test_scaled,
            y_test
        )

        threshold_results = {}

        for model_name, model in trained_models.items():
            threshold_results[model_name] = tune_threshold(
                model,
                X_test_scaled,
                y_test,
                model_name
            )

        business_results = business_capacity_analysis(
            trained_models,
            X_test_scaled,
            y_test
        )



        create_visualizations(
            trained_models,
            X_test_scaled,
            y_test
        )

        save_final_model(
            trained_models,
            scaler
        )


        print("\n" + "=" * 60)
        print("DATASET INSPECTION COMPLETED")
        print("=" * 60)

    except FileNotFoundError:

        print(
            "\nERROR: diabetes.csv was not found."
            "\nPlease check that it exists inside:"
            "\ndataset/diabetes.csv"
        )

    except Exception as error:

        print(f"\nERROR: {error}")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()