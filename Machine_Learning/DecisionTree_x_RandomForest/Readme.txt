1. Import Housing data from a dictionary format.
2. Convert the data into a DataFrame.
3. Set features and target columns.
4. Check the description and info of the data to ensure accuracy.
5. Perform data exploration.
6. Split the data, dropping any unnecessary columns.
7. Apply standard scaling using sklearn.preprocessing.StandardScaler().
   - Use StandardScaler.fit_transform() and save the transformed data into a new variable.
8. Split the dataset into training and testing sets using train_test_split(x, y, test_size=0.3, random_state=42).
   - Random state is set to 42 for consistent random selection of rows.
9. Check the shape of the training and testing datasets (x_train, x_test, etc.).
10. Implement decision tree regression.
    - Use sklearn.tree.DecisionTreeRegressor().
    - Explore the parameters of the Decision Tree if necessary.
11. Fit the decision tree regressor to the training data (dtr.fit(train_data)).
12. Predict the target variable using the test data (dtr.predict(x_test)).
    - Avoid providing y_test data for prediction; it's reserved for validation.
13. Evaluate the model's performance by checking the error.
    - Import mean_squared_error from sklearn.metrics.
    - Interpret the error as a percentage (e.g., 20.354% means 20% error).