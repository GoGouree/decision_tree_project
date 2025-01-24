# Decision Tree project

This project demonstrates a machine learning model using a Decision Tree algorithm. The model is served using FastAPI, allowing users to access it via HTTP requests.

## Machine Learning Model

The Decision Tree model is a popular supervised learning algorithm used for classification and regression tasks. It works by splitting the data into subsets based on the value of input features, creating a tree-like structure of decisions.

### Training the Model

The model is trained using a dataset that includes features and target labels. The training process involves:

1. Loading the dataset.
2. Preprocessing the data (e.g., handling missing values, encoding categorical variables).
3. Splitting the data into training and testing sets.
4. Training the Decision Tree model on the training set.
5. Evaluating the model's performance on the testing set.

## Serving the Model with FastAPI

FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints.

### Setting Up FastAPI

To serve the model using FastAPI, follow these steps:

1. **Install FastAPI and Uvicorn**:
    ```bash
    pip install fastapi uvicorn
    ```

2. **Create the FastAPI App**:
    ```python
    from fastapi import FastAPI
    import joblib

    app = FastAPI()

    # Load the trained model
    model = joblib.load("path_to_your_model.pkl")

    @app.post("/predict")
    async def predict(features: dict):
        # Preprocess the features as required by your model
        # Example: features = preprocess(features)
        prediction = model.predict([list(features.values())])
        return {"prediction": prediction[0]}
    ```

3. **Run the FastAPI App**:
    ```bash
    uvicorn main:app --reload
    ```

### Accessing the Model

Users can access the model by sending a POST request to the `/predict` endpoint with the input features in JSON format. For example:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"feature1": value1, "feature2": value2, ...}'
```

The server will respond with the model's prediction.

## Conclusion

This project showcases how to train a Decision Tree model and serve it using FastAPI. Users can easily access the model via HTTP requests to make predictions based on their input features.

For more details, refer to the project documentation and source code.
