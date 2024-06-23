# Load the attitude dataset
data("attitude")

# Construct the mode function to calculate the mode of each column
get_mode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# Loop through each column in the "attitude" dataset
for (col in names(attitude)) {
  cat("\n---", col, "---\n")
  # Measures of Central Tendency
  cat("Mean: ", mean(attitude[[col]]), "\n")
  cat("Median: ", median(attitude[[col]]), "\n")
  cat("Mode: ", get_mode(attitude[[col]]), "\n")
  
  # Measures of Variability
  cat("Max: ", max(attitude[[col]]), "\n")
  cat("Min: ", min(attitude[[col]]), "\n")
  cat("Range: ", range(attitude[[col]]), "\n")
  cat("Quantiles: ", quantile(attitude[[col]]), "\n")
  cat("IQR: ", IQR(attitude[[col]]), "\n")
  cat("Variance: ", var(attitude[[col]]), "\n")
  cat("Standard Deviation: ", sd(attitude[[col]]), "\n")
}

# Construct a Correlation Matrix 
cat("\n--- Correlation Matrix ---\n")
print(cor(attitude))

# Check the work
summary(attitude)

# Construct a multivar linear regression model
model_multivar <- lm(rating ~ complaints + privileges + learning, data = attitude)

# Check the model
summary(model_multivar)

# Explanation of Linear Regression model:The model shows a strong and 
# significant relationship between the predictors and the dependent variable, 
# with complaints being a particularly strong predictor of rating. 
# However, privileges do not seem to significantly affect the rating,
# and learning shows a positive but marginally significant relationship. 
# The overall fit of the model is good, explaining a substantial portion of the variability in rating.

# Construct a variance analysis model
model_aov <- aov(rating ~ complaints, data = attitude)

# Check the model
summary(model_aov)

# Conclusion of ANOVA model:The ANOVA table indicates that the complaints predictor is highly 
# significant in explaining the variance in the dependent variable. 
