# MLOps Assignment 1 - 8005

## Step-wise Description

This assignment trains a Random Forest regression model for house price prediction using a subset of the California Housing dataset.

As I did'nt run this from VS studio, instead I clone the Github repo in COLAB
and run these commands in a python file named MLOps_A1.ipynb

## Clone GitHub repo for accessing the project files

!git clone https://github.com/sabasajid09/A1-MLOps_8005

## Install Dependencies

!pip install -r requirements.txt

## change dorectory to your project file name i.e. A1-MLOps_8005 in my case

%cd A1-MLOps_8005

## Run the file train_8005.py to get the required model in model folder

!python src/train_8005.py

## Check the saved model in the given path with command

!ls model

model/house_price_model_8005.pkl
(The trained model will be saved in above location)

## Download and save in VS code model folder

Then add, commit and push to update Github accordingly.
