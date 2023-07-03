import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Read the data
df = pd.read_csv('game_data.csv')

# Function to calculate win/loss ratio
def calculate_ratio(x):
    if '-' in x and x.strip():
        try:
            wins, losses = x.split('-')
            return int(wins) / int(losses) if int(losses) != 0 else 0
        except ValueError:
            print(f'Error in conversion: {x}') 
            return 0
    else:
        return 0

# Generate win/loss ratio
df['win_loss_ratio'] = df['team_wins_losses_home'].apply(calculate_ratio)

# Fill missing values with mean
df.fillna(df.mean(), inplace=True)

# Define predictors and target
predictors = ['team_id_home', 'team_id_away', 'win_loss_ratio']
target = 'pts_home'

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df[predictors], df[target], test_size=0.2, random_state=0)

# Create a model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")
