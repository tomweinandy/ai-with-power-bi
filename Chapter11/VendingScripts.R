 Scripts used for Chapter 11: Advanced Power BI and AI with Data Science Languages

#################################################################################
# Ingesting Data with R
#################################################################################

# Set directory
filepath <- "C:/Users/tomwe/Documents/GoogleTrendsData/"
setwd(filepath)

# Read the list of files in the specified folder
file_list <- list.files()

# Extract "multiTimeline" files
filtered_list <- grep("^multiTimeline", file_list, value=TRUE)

# Identify the most recent date
most_recent_file <- max(filtered_list)

# Load data
full_filepath <- paste(filepath, most_recent_file, sep="")
df <- read.csv(most_recent_file, skip=2)
print(df)


#################################################################################
# Transforming Data with R
#################################################################################

# Import packages
library(zoo)

# Rename dataframe
df_rolling <- dataset

# Rename columns
colnames(df_rolling)[colnames(df_rolling ) == "Month"] <- "month_year"
colnames(df_rolling)[colnames(df_rolling ) == "python.machine.learning...United.States."] <- "python_machine_learning"
colnames(df_rolling)[colnames(df_rolling ) == "r.machine.learning...United.States."] <- "r_machine_learning"

# Add six-month moving average
df_rolling$month_year <- as.Date(paste(df_rolling$month_year, "01", sep = "-"))
df_rolling$python_moving_average <- rollapply(df_rolling$"python_machine_learning", width=6, FUN=mean, align="right", partial=TRUE)
df_rolling$r_moving_average <- rollapply(df_rolling$r_machine_learning, width=6, FUN=mean, align="right", partial=TRUE)

# Load data
print(df_rolling)


#################################################################################
# Visualizing Data with R
#################################################################################

# Import packages
library(ggplot2)

# Encode data to proper format (not necessary in R Studio)
dataset$month_year <- as.Date(dataset$month_year)
dataset$python_machine_learing <- as.numeric(dataset$python_machine_learning)
dataset$r_machine_learning <- as.numeric(dataset$r_machine_learning)
dataset$python_moving_average <- as.numeric(dataset$python_moving_average)
dataset$r_moving_average <- as.numeric(dataset$r_moving_average)

# Display plot
ggplot(dataset, aes(x=month_year)) +

  # Add scatter plots
  geom_point(aes(y=python_machine_learning, color="gold"), alpha=0.5, size=4) +
  geom_point(aes(y=r_machine_learning, color="red"), alpha=0.5, size=4) +

  # Add line plots
  geom_line(aes(y=python_moving_average, color="gold"), lwd=2) +
  geom_line(aes(y=r_moving_average, color="red"), lwd=2) +

  # Add labels
  labs(x = "", y = "Google Search Frequency Index") +
  ggtitle("Change in Programming Language Popularity Over Time") +

  # Format
  theme_minimal() +
  theme(text=element_text(size=24)) +
  theme(plot.title = element_text(hjust = 0.5),
        panel.grid = element_blank(),
        panel.border = element_rect(color = "black", fill=NA, size = 0.5),
        legend.position = c(0.15, 0.9),
        legend.background = element_blank()) +

  # Override legend so colors, labels align with plot
  scale_color_identity(name="",
                       breaks=c("gold", "red"),
                       labels=c("Python machine learning", "R machine learning"),
                       guide = "legend")


#################################################################################
# Training an Model with R on Ingest
#################################################################################

# Load packages
library(readr)
library(xgboost)

# Load CSV from GitHub
url <- "https://raw.githubusercontent.com/tomweinandy/AIinPowerBI/main/vending_revenue.csv"
df <- read.csv(url)

# Clean data
df <- na.omit(df)
df <- subset(df, select = -c(WeekStartingSat))
df$Location <- as.factor(df$Location)

# Split the data into training and validation sets
set.seed(24)
train_indices <- sample(1:nrow(df), 0.7 * nrow(df))
train_data <- df[train_indices, ]
test_data <- df[-train_indices, ]

# Define the features from the target variable "Location"
features <- setdiff(names(train_data), "Location")

# Convert the data to DMatrix format (an efficient data structure used by xgboost)
dtrain <- xgb.DMatrix(data = as.matrix(train_data[, features]), label = train_data$Location)
dtest <- xgb.DMatrix(data = as.matrix(test_data[, features]), label = test_data$Location)

# Set hyperparameters for the xgboost model
params <- list(
  objective = "multi:softprob", # For multi-class classification problems
  num_class = length(levels(df$Location)), # Number of classes in the target variable
  eta = 0.1, # Learning rate
  max_depth = 6, # Maximum depth of each tree
  nrounds = 100 # Number of boosting rounds
)

# Train the model
model <- xgboost(params, data = dtrain, nrounds = params$nrounds)

# Shuffle and encode original dataset
df_shuffled = df[sample(1:nrow(df)), ]
df_shuffled_matrix = xgb.DMatrix(data = as.matrix(df_shuffled[, features]), label = df_shuffled$Location)

# Make prediction using trained model
final_predictions <- predict(model, df_shuffled_matrix)

# Convert the predicted probabilities to class labels
final_predicted_ints = as.integer(round(final_predictions))
location_mapping <- c('Factory', 'Library', 'Mall 1', 'Mall 2', 'Office')
final_predicted_classes <- factor(final_predicted_ints, levels = 1:5, labels = location_mapping)

# Add predicted column to original cleaned dataframe
df_final = cbind(PredictedLocation = final_predicted_classes, df_shuffled)
