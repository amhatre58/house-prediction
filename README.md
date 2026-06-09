## House Price Prediction Using Tree-Based Models
This project implements machine learning regression systems to predict property prices based on key features such as size, location, bedroom count, and house age. It highlights how tree-based models capture complex, non-linear relationships to deliver accurate valuations in the real estate market.
------------------------------
## 📌 Project Overview
Real estate pricing is inherently non-linear; the value of a house does not increase in a perfectly straight line with size or age. This project demonstrates how tree-based algorithms handle these structural complexities without requiring strict statistical assumptions, serving as an automated tool for buyers, sellers, and investors.
------------------------------
## ⚙️ Core Workflow## 1. Feature Engineering & Selection

* Categorical Encoding: Converting location parameters and neighborhood data into machine-readable formats using Target Encoding or One-Hot Encoding.
* Feature Creation: Calculating secondary metrics like property age from the build year to better capture depreciation.
* Handling Multicollinearity: Evaluating features to ensure redundant variables do not negatively impact tree splitting metrics.

## 2. Model Training & Exploration
The project builds and compares the following regression architectures:

* Decision Tree Regressor: Establishing a structural baseline to map out hierarchical, conditional splits.
* Random Forest Regressor: Utilizing an ensemble of independent trees (bagging) to reduce overfitting and variance.
* Hyperparameter Tuning: Optimizing parameters like max_depth, min_samples_split, and n_estimators to maximize generalization.

## 3. Feature Importance Analysis

* MDI (Mean Decrease in Impurity): Computing native feature importance scores directly from the Random Forest model.
* Permutation Importance: Evaluating how much prediction error increases when a specific feature's values are shuffled.
* Insights Mapping: Visualizing which characteristics (e.g., location vs. square footage) exert the strongest mathematical pull on final market prices.

------------------------------
## 📊 Evaluation Metrics
Model accuracy and error margins are tracked and benchmarked using:

* Root Mean Squared Error (RMSE): Measuring the average magnitude of prediction errors, penalizing larger deviations.
* Mean Absolute Error (MAE): Measuring the average absolute distance between predicted and actual property values.
* R-squared (R²): Quantifying the proportion of variance in housing prices explained by the trained model features.

------------------------------
