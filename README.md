# Machine Learning Fundamentals - Cumulative Lab

## Introduction

In this cumulative lab, I work through an end-to-end machine learning workflow, focusing on the fundamental concepts of machine learning theory and processes. The main emphasis is on modeling theory (not EDA or preprocessing), so we will skip over some of the data visualization and data preparation steps that one would take in an actual modeling process.

## Objectives

* Recall the purpose of, and practice performing, a train-test split
* Recall the difference between bias and variance
* Practice identifying bias and variance in model performance
* Practice applying strategies to minimize bias and variance
* Practice selecting a final model and evaluating it on a holdout set

## Our Task: Build a Model to Predict Blood Pressure

![stethoscope sitting on a case](images/stethoscope.jpg)

<span>Photo by <a href="https://unsplash.com/@marceloleal80?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Marcelo Leal</a> on <a href="https://unsplash.com/s/photos/blood-pressure?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>

### Business and Data Understanding

Hypertension (high blood pressure) is a treatable condition, but measuring blood pressure requires specialized equipment that most people do not have at home.

The question, then, is ***can we predict blood pressure using just a scale and a tape measure***? These measuring tools, which individuals are more likely to have at home, might be able to flag individuals with an increased risk of hypertension.

[Researchers in Brazil](https://doi.org/10.1155/2014/637635) collected data from several hundred college students in order to answer this question. We will be specifically using the data they collected from female students.

The measurements we have are:

* Age (age in years)
* BMI (body mass index, a ratio of weight to height)
* WC (waist circumference in centimeters)
* HC (hip circumference in centimeters)
* WHR (waist-hip ratio)
* SBP (systolic blood pressure)

The chart below describes various blood pressure values:

<a title="Ian Furst, CC BY-SA 4.0 &lt;https://creativecommons.org/licenses/by-sa/4.0&gt;, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Hypertension_ranges_chart.png"><img width="512" alt="Hypertension ranges chart" src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Hypertension_ranges_chart.png/512px-Hypertension_ranges_chart.png"></a>

### Steps

#### 1. Perform a Train-Test Split

First we will load the data into a dataframe using pandas, separate the features (`X`) from the target (`y`), and use the `train_test_split` function to separate data into training and test sets.

#### 2. Build and Evaluate a First Simple Model

Using the `LinearRegression` model and `mean_squared_error` function from scikit-learn, we will build and evaluate a simple linear regression model using the training data. We'll also use `cross_val_score` to simulate unseen data, without actually using the holdout test set.

#### 3. Use `PolynomialFeatures` to Reduce Underfitting

We will apply a `PolynomialFeatures` transformer to give the model more ability to pick up on information from the training data. We'll then test out different polynomial degrees until we have a model that is perfectly fit to the training data.

#### 4. Use Regularization to Reduce Overfitting

Instead of a basic `LinearRegression`, we'll use a `Ridge` regression model to apply regularization to the overfit model. In order to do this we will need to scale the data. We'll then test out different regularization penalties to find the best model.

#### 5. Evaluate a Final Model on the Test Set

We'll preprocess `X_test` and `y_test` appropriately in order to evaluate the performance of our final model on unseen data.


