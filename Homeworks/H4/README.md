Add your forth homework as a pull request to this folder.

Deadline 2020-04-16 EOD


Task:
For a selected data set (you can use data from your project or data from Homework 1) prepare a knitr/jupiter notebook with the following points.
Submit your results on GitHub to the directory Homeworks/H4.

TODO:

1. For the selected data set, train at least one tree-based ensemble model (random forest, gbm, catboost or any other boosting)
2. for some selected observation from this dataset, calculate the model predictions for model (1)
3. for an observation selected in (2), calculate the decomposition of model prediction using Ceteris paribus / ICE profiles (packages for R: DALEX, ALEPlot, ingredients, packages for python: pyCeterisParibus).
4. find two observations in the data set, such that they have different CP profiles (e.g. model response is growing with age for one observations and lowering with age for another). Note that you need to have model with interactions to have such differences
5. train a second model (of any class, neural nets, linear, other boosting) and find an observation for which CP profiles are different between the models
6. Comment on the results for points (4) and (5)


**Important note:**

The submitted homework should be in html format (generated from a knitter/jupiter) and should consist of two parts. 

The first part is the key results and comments from points 4-6. In this part **PLESE DO NOT SHOW ANY R/PYTHON CODES, RESULTS (IMAGES, COMMENTS) ONLY.**

The second part should start with the word Appendix or Załącznik and should include the reproducible R/PYTHON code used to implement points 1-6.

Such division 1. will make these homework more readable, 2. will create good habits related to reporting.

