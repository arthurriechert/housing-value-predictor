# Housing Value Predictor
This is the culmination of my learning from Andrew Ng's first 3-week course on machine learning.

## Plans
I will update this regularly as I learn more information.

## Description
Implements a simple polynomial regression model to predict the value of a house according to seven features

It uses Z-score normalization due to the wide range of data value.

Uses traditional gradient descent.

Includes easy-to-use CLI to get novel predictions.

### Features
1) Longitude
2) Latitude
3) Longitutde * Latitude
4) Median Age
5) Total Rooms
6) Median Income
7) Total Bedrooms
8) Population

## Note
Project still needs many adjustments as it tends to perform poorly on novel data.

### Output
Housing Value in USD

# Data Set
The data set is from the California area so may not be generalizable to other regions.

Link: https://www.kaggle.com/datasets/walacedatasci/hands-on-machine-learning-housing-dataset

# Setup
1) Clone the repository

2) Cd into repo

3) Run: python3.11 -m venv env

4) Run: source env/bin/activate

5) Run: pip install -r requirements.txt

6) Cd into src/

7) Run: python main.py
