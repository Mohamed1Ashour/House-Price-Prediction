# House Price Prediction
This is a machine learning project that predicts the sale price of a house based on various features such as square footage, number of bedrooms, and number of bathrooms. The project consists of a Jupyter Notebook that trains a machine learning model on a dataset of house prices.

# Getting Started
To run the project, you will need to install Python 3 and several Python packages, including NumPy and Scikit-Learn. You can install these packages using pip, the package installer for Python:

pip install numpy scikit-learn
You will also need to download the dataset of house prices, which is available in the "data" folder of this repository.

# Training the Model
To train the machine learning model, open the "house_price_prediction.ipynb" Jupyter Notebook in a Jupyter Notebook server. The notebook contains detailed instructions for loading the dataset, preprocessing the data, and training the model using the Scikit-Learn library.

Once you have trained the model, you can save it using the pickle module in Python:

python
import pickle

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
The pickled model can then be used to make predictions in other Python scripts or applications.

# Making Predictions
To make predictions using the trained machine learning model, you can load the pickled model using the pickle module in Python:

python
import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
You can then use the predict method of the model object to make predictions on new data:

python
import numpy as np

#Input data: square footage, number of bedrooms, number of bathrooms
X_test = np.array([[1500, 3, 2], [2000, 4, 3]])

#Make predictions
y_pred = model.predict(X_test)

#Print predicted prices
print(y_pred)
The predict method will return an array of predicted prices for the input data.

# Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request. We welcome contributions of any kind, including bug fixes, feature requests, and documentation improvements.

# License
This project is licensed under the MIT License. See the LICENSE file for more information.
