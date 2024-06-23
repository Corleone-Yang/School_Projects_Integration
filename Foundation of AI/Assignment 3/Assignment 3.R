# Task 1

# import library
library(MASS)
library(ggplot2)
library(tibble)

# a)

Cars93 <- as_tibble(Cars93)
qplot(data = Cars93, x = EngineSize, y = Price, geom = "point")

# a. Relationship: Price and EngineSize have a Positive Correlation relationship.

# b)
ggplot(data = Cars93, aes(x = EngineSize, y = Price)) + geom_point()

# c)
ggplot(data = Cars93, aes(x = EngineSize, y = Price, color = Type, shape = DriveTrain)) + geom_point()

# a. 
# Compact and Small: These cars cluster towards the lower end of both engine size and price.
# Midsize and Large : These cars cluster towards the moderate range fo engine sizes and prices
# Sporty and Van: These cars cluster towards the larger engine size and higher price range.

# b.
# 4WD: These vehicles might be found with a range of engine sizes but could be associated with higher prices due to the additional complexity of the four-wheel-drive system.
# Front: Front-wheel drive cars are common in compact and midsize segments and might cluster towards the lower to mid-range of engine sizes and prices.
# Rear: Rear-wheel drive is often found in sports and luxury cars, which may lead to a cluster of higher engine sizes and prices.

# d)
ggplot(data = Cars93, aes(x = AirBags, y = Price)) + geom_boxplot()

# a. Cars with Driver & Passenger airbags seem to have a higher median price than the other categories. 
# The Driver only category has a lower median price than the Driver & Passenger category but is higher than the None category.
# Cars with No airbags have the lowest median price.

# Task 2

# Import library
library(maps)
library(ggplot2)
library(dplyr)

# Load the USArrests data
data("USArrests")

# Convert the row names of USArrests to a column
USArrests$region <- tolower(rownames(USArrests))

# Get the map data for US states
states_map <- map_data("state")

# Merge the USArrests data with the state map data
merged_data <- merge(states_map, USArrests, by = "region")

# Create the heat map
ggplot(merged_data, aes(x = long, y = lat, group = group, fill = Assault)) +
  geom_polygon(color = "white") +
  coord_fixed(1.4) +
  theme_void() +
  theme(panel.background = element_rect(fill = "lightgray")) +
  labs(fill = "Assault Rates") +
  scale_fill_gradient(low = "lightblue", high = "red")

# Task 3
# Import library
library(gridExtra)

# Load the data
setwd("/Users/mac/Desktop/Assignment\ 3")
df <- read.csv("summer_winter_olympics.csv")

# a)
p1 <- ggplot(data = df, aes(x = X..Summer)) +
  geom_histogram(binwidth = 1, fill = "yellow", color = "black") +
  geom_text(stat = 'count', aes(label = after_stat(count), y = after_stat(count)), vjust = -0.5, color = "black") +
  labs(x = "Total Summer Games", y = "Frequency", title = "Histogram of Summer Games")

# b)
p2 <- ggplot(data = df, aes(x = X..Winter)) +
  geom_histogram(binwidth = 1, fill = "blue", color = "black") +
  geom_text(stat = 'count', aes(label = after_stat(count), y = after_stat(count)), vjust = -0.5, color = "black") +
  labs(x = "Total Winter Games", y = "Frequency", title = "Histogram of Winter Games")

# c)
grid.arrange(p1, p2, ncol = 2)

# d)
p3 <- ggplot(data = df, aes(x = Total)) +
  geom_histogram(binwidth = 100, fill = "yellow", color = "black") +
  labs(x = "Total Medals in Summer Games", y = "Frequency", title = "Histogram of Summer Games Medals")
p4 <- ggplot(data = df, aes(x = Total.1)) +
  geom_histogram(binwidth = 10, fill = "blue", color = "black") +
  labs(x = "Total Medals in Winter Games", y = "Frequency", title = "Histogram of Winter Games Medals")
grid.arrange(p3, p4, ncol = 2)

# e) Total Summer Medals and Total Winter Medals have a positive correlation
ggplot(df, aes(x=Total, y=Total.1)) +
  geom_point(aes(color=Combined.total), size=3) +
  labs(x="Total Summer Medals", y="Total Winter Medals", color="Combined Total Medals") +
  theme_minimal()

# f)
ggplot(df, aes(x=X..Summer, y=X..Winter)) +
  geom_point(aes(color=Combined.total), size=3) +
  labs(x="Number of Summer Games", y="Number of Winter Games", color="Combined Total Medals") +
  theme_minimal()

# g)
p5 <- ggplot(df, aes(x=X)) + geom_histogram(binwidth=20, fill="gold") + labs(title="Summer Gold Medals", x="Number of Gold Medals", y="Count") + theme_minimal()
p6 <- ggplot(df, aes(x=X.1)) + geom_histogram(binwidth=20, fill="#C0C0C0") + labs(title="Summer Silver Medals", x="Number of Silver Medals", y="Count") + theme_minimal()
p7 <- ggplot(df, aes(x=X.2)) + geom_histogram(binwidth=20, fill="brown") + labs(title="Summer Bronze Medals", x="Number of Bronze Medals", y="Count") + theme_minimal()
p8 <- ggplot(df, aes(x=X.3)) + geom_histogram(binwidth=20, fill="gold") + labs(title="Winter Gold Medals", x="Number of Gold Medals", y="Count") + theme_minimal()
p9 <- ggplot(df, aes(x=X.4)) + geom_histogram(binwidth=20, fill="#C0C0C0") + labs(title="Winter Silver Medals", x="Number of Silver Medals", y="Count") + theme_minimal()
p10 <- ggplot(df, aes(x=X.5)) + geom_histogram(binwidth=20, fill="brown") + labs(title="Winter Bronze Medals", x="Number of Bronze Medals", y="Count") + theme_minimal()

grid.arrange(p5, p6, p7, p8, p9, p10, ncol=3)

# h)
p11 <- ggplot(df, aes(x=X)) + geom_histogram(binwidth=10, fill="gold") + labs(title="Summer Gold Medals", x="Number of Gold Medals", y="Count") + theme_minimal()
p12 <- ggplot(df, aes(x=X.1)) + geom_histogram(binwidth=10, fill="#C0C0C0") + labs(title="Summer Silver Medals", x="Number of Silver Medals", y="Count") + theme_minimal()
p13 <- ggplot(df, aes(x=X.2)) + geom_histogram(binwidth=10, fill="brown") + labs(title="Summer Bronze Medals", x="Number of Bronze Medals", y="Count") + theme_minimal()
p14 <- ggplot(df, aes(x=X.3)) + geom_histogram(binwidth=10, fill="gold") + labs(title="Winter Gold Medals", x="Number of Gold Medals", y="Count") + theme_minimal()
p15 <- ggplot(df, aes(x=X.4)) + geom_histogram(binwidth=10, fill="#C0C0C0") + labs(title="Winter Silver Medals", x="Number of Silver Medals", y="Count") + theme_minimal()
p16 <- ggplot(df, aes(x=X.5)) + geom_histogram(binwidth=10, fill="brown") + labs(title="Winter Bronze Medals", x="Number of Bronze Medals", y="Count") + theme_minimal()

grid.arrange(p11, p12, p13, p14, p15, p16, ncol=3)

# i) Top 10 countries have more medals than the sum of rest countries
