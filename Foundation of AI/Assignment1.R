# Yahe_Yang Jan/26/2024

# read the data of csv format into a data frame
Federal_Stimulus_Data <- read.csv("/Users/mac/Desktop/mydata/Use_of_ARRA_Stimulus_Funds_20240125.csv")

# change the name of data frame
fed_stimulus <- Federal_Stimulus_Data

# remove the original data frame
rm(Federal_Stimulus_Data)

# compute sum
total_payment <- sum(fed_stimulus$Payment.Value, na.rm = TRUE)

#compute mean
average_payment <- mean(fed_stimulus$Payment.Value, na.rm = TRUE)

# print the outcomes
print(paste("Total Payment: ", total_payment))
print(paste("Average Payment: ", average_payment))

# Create subset where Project.Status == "Completed 50% or more"
subset_fed_stimulus <- subset(fed_stimulus, Project.Status == "Completed 50% or more")

# Create a knitr report
#install.packages("knitr")
#install.packages("rmarkdown")
# Create the report process “File” -> “New File” -> “R Markdown...”
