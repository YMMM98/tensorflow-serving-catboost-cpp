#!/usr/bin/env python3
# Export a test CatBoost model for testing purposes

import catboost
from catboost import CatBoostClassifier, Pool
import numpy as np
import os

def main():
    # Create simple synthetic data
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

    # Train a simple CatBoost model
    model = CatBoostClassifier(
        iterations=10,
        depth=4,
        learning_rate=0.1,
        loss_function='Logloss',
        verbose=False
    )

    model.fit(X_train, y_train)

    # Save the model
    output_dir = os.path.join(
        os.path.dirname(__file__),
        'test_model',
        '1'
    )
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, 'catboost.cbm')
    model.save_model(model_path, format='cbm')

    print(f"Model saved to: {model_path}")

    # Test prediction
    X_test = np.random.randn(2, 10)
    predictions = model.predict(X_test)
    print(f"Test predictions: {predictions}")

if __name__ == "__main__":
    main()
