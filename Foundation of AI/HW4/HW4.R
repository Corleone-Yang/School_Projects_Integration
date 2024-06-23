library(readxl)
library(rpart)
library(rpart.plot)
library(e1071)
library(caret)
library(ggplot2)
library(neuralnet)
library(dplyr)
library(fastDummies)
library(cluster)

# Load the data
titanic_data <- read_excel("/Users/mac/Desktop/Titanic_Data.xls")
# Check the head of data
head(titanic_data)

# Task 1: Decision Tree
titanic_data_task1 <- titanic_data[]

# Focus on certain columns (I don't pick Life Boat because of too much missing data)
titanic_data_task1 <- titanic_data[, c("Survived", "Passenger Class", "Sex", "Age", "Passenger Fare", "Port of Embarkation")]

# Check the head of data
head(titanic_data_task1)
# a)
#Convert the Sex variable to numeric
titanic_data_task1$Sex <- as.numeric((factor(titanic_data_task1$Sex)))

# Split data into train set and test set
set.seed(42)  # For reproducibility
index <- createDataPartition(titanic_data_task1$Survived, p = 0.8, list = FALSE)
train_data <- titanic_data_task1[index, ]
test_data <- titanic_data_task1[-index, ]

# Fit the decision tree model
decision_tree_model <- rpart(Survived ~ `Passenger Class` + Sex + Age + `Passenger Fare` + `Port of Embarkation`, data = train_data, method = "class")

# Evaluate the model
predictions <- predict(decision_tree_model, test_data, type = "class")
accuracy <- sum(predictions == test_data$Survived) / nrow(test_data)
print(paste("Accuracy:", accuracy))

# Visualize
plot(decision_tree_model, uniform=TRUE, main="Decision Tree")
text(decision_tree_model, use.n=TRUE, all=TRUE, cex=.8)

# b)
# target variable: Survived

# c)
# key attributes: Sex, Age, Passenger Class, Passenger Fare. These attributes are selected based on their importance 
# in the model's ability to accurately predict the target variable, as determined by the algorithm during the model training
# process. The splits on these attributes indicate the decision points that the model uses to classify a passenger as survived or not survived.

# d)
# The best depth for my model is 2 for left subtree and 4 for right subtree. It is because that after the certain depth,
# the accuracy of my model doesn't have a obvious improvement.


# Task2: Support Vector Machine(SVM)
# a)
titanic_data_task2 <- titanic_data

# Selecting numeric variables, and keep the survived column
Survived <- titanic_data$Survived
titanic_data_numeric <- titanic_data[, sapply(titanic_data, is.numeric)]
titanic_data_task2 <- cbind(Survived, titanic_data_numeric)

# b)
# Replace missing values with the mean of the respective column
titanic_data_task2 <- data.frame(lapply(titanic_data_task2, function(x) {
  if (is.numeric(x)) {
    x[is.na(x)] <- mean(x, na.rm = TRUE)
  }
  return(x)
}))

# Split the data into train set and test set
set.seed(42)  # for reproducibility
sample_index <- sample(1:nrow(titanic_data_task2), 0.8 * nrow(titanic_data_task2))
train_data <- titanic_data_task2[sample_index, ]
test_data <- titanic_data_task2[-sample_index, ]

# Build the SVM model
train_data$Survived <- as.factor(train_data$Survived)
svm_model <- svm(Survived ~ ., data = train_data, kernel = "linear")

# Evaluate the model
predictions <- predict(svm_model, test_data)
table(predictions, test_data$Survived)
confusion_matrix <- table(predictions, test_data$Survived)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(accuracy)

# c) Visualize
# Get the model coefficients for the support vectors
sv_coefs <- svm_model$coefs
sv_data <- svm_model$SV

# Calculate the weights
# In a linear SVM, the weights (w) are obtained by w = t(coefs) %*% SVs
weights <- t(sv_coefs) %*% sv_data

# Since we're dealing with a binary classification, there will be only one row of weights.
# We need to transpose and convert it to a numeric vector.
weight_vector <- as.numeric(weights)

# Match the weight vector with the feature names
# We skip the first column since it's the 'Survived' column, which is our response variable.
feature_names <- names(train_data)[-1]

# Create a data frame for plotting
weight_df <- data.frame(Feature = feature_names, Weight = weight_vector)

# Sort the weights for better visualization
weight_df <- weight_df[order(abs(weight_df$Weight), decreasing = TRUE), ]

# Plot the weights using ggplot2
ggplot(weight_df, aes(x = reorder(Feature, Weight), y = Weight)) +
  geom_col() +
  coord_flip() +  # Flip the axes for horizontal bars
  theme_minimal() +
  xlab("Feature") +
  ylab("Weight") +
  ggtitle("Feature Weights in SVM Model")

# Task 3
# a)
logistic_model <- glm(Survived ~ ., data = train_data, family = binomial)

# b)
# Assuming you have a logistic regression model named logistic_model
coefficients <- summary(logistic_model)$coefficients
model_coefficients <- coef(logistic_model)

# Create the model equation as a string
equation <- paste0("log(p / (1 - p)) = ", round(coefficients[1], 2))
for (i in 2:length(coefficients)) {
  coef_value <- round(coefficients[i], 2)
  equation <- paste0(equation, " + ", coef_value, " * X", i-1)
}

# Print the model equation
equation

# Visualize# Create a dataframe for plotting
coefficients_df <- data.frame(
  Feature = names(model_coefficients),
  Coefficient = model_coefficients
)

ggplot(coefficients_df, aes(x = Feature, y = Coefficient)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  theme_minimal() +
  coord_flip() +  # Flip the axes for horizontal bar chart
  xlab("Features") +
  ylab("Coefficients") +
  ggtitle("Logistic Regression Model Coefficients")

# c)
# The No.of.Siblings.or.Spouses.On.Board is the most important variable to estimate my target variable.
# It is because that the absolute value of Passenger.Fare is the biggest among all variables.

# Task 4
# a)
titanic_data_task4 <- titanic_data_task2

# Assuming 'Survived' is the target variable and it's a factor
titanic_data_task4$Survived <- as.factor(titanic_data_task4$Survived)

set.seed(42)  # for reproducibility
sample_size <- floor(0.8 * nrow(titanic_data_task4))
sample_index <- sample(1:nrow(titanic_data_task4), sample_size, replace = FALSE)
train_data <- titanic_data_task4[sample_index, ]
test_data <- titanic_data_task4[-sample_index, ]


# Train the ANN model
nn_formula <- Survived ~ Age + No.of.Siblings.or.Spouses.on.Board + No.of.Parents.or.Children.on.Board + Passenger.Fare

# Train the model
nn_model <- neuralnet(nn_formula, data = train_data, hidden = c(5), linear.output = FALSE, act.fct = "tanh")

# Check the result
summary(nn_model)

# b)
# Input features: 'Age', 'No.of.Siblings.or.Spouses.on.Board', 'No.of.Parents.or.Children.on.Board', and 'Passenger.Fare'
# Hidden Layer: 5 neurons
# Output: 2. Due to this is a binary classification

# Task 5
# a)
titanic_data_task5 <- titanic_data

titanic_data_task5 <- titanic_data_task5 %>% 
  select(Age, `Passenger Fare`) %>%
  na.omit() # Remove rows with missing values

# Standardize the data
titanic_data_scaled <- scale(titanic_data_task5)

# Perform K-means clustering with different values of k
set.seed(42) # Set seed for reproducibility
k_values <- 2:5
for (k in k_values) {
  kmeans_result <- kmeans(titanic_data_scaled, centers = k, nstart = 25)
  cat("For k =", k, ", the total within-cluster sum of squares is", 
      round(kmeans_result$tot.withinss, 2), "\n")
}