# Stock Prediction Using SimpleRNN & LSTM
RNN stands for Recurrent Neural Networks. RNN is used when we have sequential data.
`SimpleRNN` is a Function that is provided by `Tensorflow`, `Keras`. It is a fully-connected RNN where the output is fed back as input.
RNN is used for **Time Series Forecasting**.
Long Short-Term Memory (LSTM) model with TensorFlow.It includes steps for loading data, preprocessing, model creation, training, evaluation, and visualization of the predictions compared to actual stock prices.

## Project objective
The objective of the Project is to execute an RNN model & LSTM Model inorder to **predict the performance of Tesla Stocks**.

## Methods
* Deep Learning Neural Networks
* Simple RNN by `Tensorflow`
* Visualizations
* Feature Engineering
* Data Normalization

## Technologies used
* Python
* Pandas
* Tensorflow
* Keras
* Scikit-learn
* Matplotlib

## Project Description
The project aims to predict the movement of Tesla Stock. 
Recurrent neural networks have the feature of taking the **output of the same artificial neuron as the input again**. The Neural Network takes the first 50 observations and then uses regression to predict the 51st observation. 
The process continues for predicting the 52nd observation and so on. This method is also called **Autoregression**.

The output-input loop feature of RNN model and Autoregression process used help in predicting the future values.

## Procedure
* The required python modules are imported.
* The data is imported as a **Data Frame using Pandas**.
* The data is separated in to **Traing set and Validation set**.
* The data from both Training as well as Validation set is Reshaped.
* The Data is Normalised using `MinMaxScaler` from Scikitlearn.
* **Feature Engineering** is done according to the model's needs. The data is segregated in order use 50 observations to predict 51st observation.
* The data is Reshaped again.

### RNN Model Building
* `Tensorflow`, `Keras` is used for building the Neural Network.
* `SimpleRNN`, `Dropout`, `Dense` layers are added to `Sequential`.

* #  LSTM model Building
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(100, 1)),  # First LSTM layer
    tf.keras.layers.LSTM(50, return_sequences=False),  # Second LSTM layer
    tf.keras.layers.Dense(25),  # Dense layer with 25 neurons
    tf.keras.layers.Dense(1)  # Output layer with 1 neuron (prediction)
])

* The model is `compiled` using
    + `'adam'` optimizer
    - `'mean_squared_error'` loss
    + `['accuracy']` metrics

* Model is then summarized
* The model is fit to Training data.
* Loss and Accuracy is plotted.
* X_train is predicted using `predict` then denormalized using `inverse_transform`.
* The predicted values are plotted along side actual values
* The process is repeated for validation set and the results are plotted.



