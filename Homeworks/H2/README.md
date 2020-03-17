Add your second homework as a pull request to this folder.

Deadline 2020-03-26 EOD


Task:
For a selected data set (you can use data from your project or data from Homework 1) prepare a knitr/jupiter notebook with the following points.
Submit your results on GitHub to the directory Homeworks/H2.

TODO:

1. For the selected data set, train at least one tree-based ensemble model (random forest, gbm, catboost or any other boosting)
2. for some selected observation from this dataset, calculate the model predictions for model (1)
3. for an observation selected in (2), calculate the decomposition of model prediction using SHAP, Break Down or both (packages for R: DALEX, iml, packages for python: shap, dalex, piBreakDown).
4. find two observations in the data set, such that they have diffeerent most important vareiables (e.g. age and gender are the most important for observation A, but race and class for observation B)
5. select one variable and find two observations in the data set such that for one observation this variable has a positive effect and for the other a negative effect
6. train a second model (of any class, neural nets, linear, other boosting) and find an observation for which BD/shap attributions are different between the models
7. Comment on the results for points (4), (5) and (6)

