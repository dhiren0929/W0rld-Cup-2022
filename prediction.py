## ---------------- Start Time ----------------
from time import time
start_time = time() 

## ---------------- Dependencies ----------------
### --- general ---
import pandas as pd 
import numpy as np
from datetime import datetime
from dateutil.parser import parse

### --- sklearn/keras <> ML ---
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense


## ---------------- Py Warnings ----------------
pd.options.mode.chained_assignment = None ## SettingWithCopyWarning

## ---------------- Data Cleaning ----------------

### --- bringing in data ---
all_matches = pd.read_csv("results.csv")
world_cup_winners = pd.read_csv("World Cup_Winners.csv")

### --- creating smaller data set for testing purposes ---
reduced_matches = all_matches[0-50:]

### --- function to make dfs more descriptive ---
def winners(df):
	### --- empty lists to be filled with data to become new columns ---
	da_winners = [] # all match winners
	da_losers = [] # all match losers
	winners_score = [] # the score of the match
	score_difference = [] # score difference
	penalties = [] # if the match went to penalties

	### --- filling the empty lists ---
	for index, row in df.iterrows():
		if row['home_score'] > row['away_score']:
			da_winners.append(row['home_team'])
			da_losers.append(row['away_team'])
			winners_score.append(row['home_score'])
		else:
			da_winners.append(row['away_team'])
			da_losers.append(row['home_team'])
			winners_score.append(row['away_score'])

		score_difference.append(abs(row['home_score'] - row['away_score']))
		if abs(row['home_score'] - row['away_score']) == 0:
			penalties.append(True)
		else:
			penalties.append(False)

	### --- adding columns to the df ---
	df["winner"] = [winner for winner in da_winners]
	df["loser"] = [loser for loser in da_losers]
	df["winners_score"] = [ws for ws in winners_score]
	df["score_difference"] = [diff for diff in score_difference]
	df["penalties"] = [p for p in penalties]

	df.iloc[18426, df.columns.get_loc('winner')] = 'Brazil'
	df.iloc[18426, df.columns.get_loc('loser')] = 'Italy'

	df.iloc[28423, df.columns.get_loc('winner')] = 'Italy'
	df.iloc[28423, df.columns.get_loc('loser')] = 'France'

	### --- stripping all object columns to ensure no spaces before or after strings ---
	### --- very time and cpu intensive ---
	'''
	column_names = df.columns
	for name in column_names:
		if df[name].dtype == object:
			for pd_index, pd_object in enumerate(df[name]):
				df[name][pd_index] = pd_object.strip()
	'''

	return df


### --- running function on the pd.dfs ---
winners(all_matches)
# print(world_cup_winners.columns)
# print()
# print(all_matches.columns)

### --- creates pd.Series of all 20 of the world cup winners and runner ups ---
just_winners = world_cup_winners["Winners"]
runner_ups = world_cup_winners["Runners-up"]
wc_years = world_cup_winners["Year"]

### --- stripping spaces ---
for i, w in enumerate(just_winners):
	just_winners[i] = w.strip()

for ii, losers in enumerate(runner_ups):
	runner_ups[ii] = losers.strip()

### --- creating the ml y, boolean column to predict wins ---
world_cup_final_game = []
counter = 0
for indx, games in all_matches.iterrows():
	if games["winner"].strip() == just_winners[counter] and games["loser"].strip() == runner_ups[counter] and str(wc_years[counter]) in games["date"]:
		world_cup_final_game.append(True)
		# print(f"the date: {games['date']} the winner: {games['winner']} & the loser: {games['loser']}")
		if counter != 20:
			counter += 1
	else:
		world_cup_final_game.append(False)

all_matches["world cup winner"] = [match for match in world_cup_final_game]

# for dx, date in enumerate(all_matches['date']):
# 	all_matches['date'][dx] = datetime.strptime(date, '%Y-%m-%d')

# print(all_matches.iloc[28400:28450,:])

print(all_matches.columns)


## ---------------- Neural Network ----------------

### --- shaping the data ---
y = all_matches["world cup winner"]
# y = y.as_matrix(columns=None).reshape(-1, 1)
y = y.values.reshape(-1, 1)
print(y.shape)
print()

X = all_matches[['home_team', 'away_team', 'home_score', 'away_score',
       'tournament', 'city', 'country', 'neutral', 'winner', 'loser',
       'winners_score', 'score_difference', 'penalties']]
X.values
print(X.shape)
# print(X.dtypes)

label_encoder = LabelEncoder()
label_encoder.fit(y)
encoded_y = label_encoder.transform(y)
one_hot_y = to_categorical(encoded_y)
# print(one_hot_y)

model = LinearRegression()
model.fit(X, y)

X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
y_train_scaled = y_scaler.transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

predictions = model.predict(X_test_scaled)
MSE = mean_squared_error(y_test_scaled, predictions)
r2 = model.score(X_test_scaled, y_test_scaled)

print(f"MSE: {MSE}, R2: {r2}")



'''
### --- creating training and testing datasets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

### --- data preprocessing; scaling the data ---
X_scaler = StandardScaler().fit(X_train)
# X_scaler = preprocessing.PowerTransformer(method='box-cox', standardize=True).fit(X_train)

X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

### --- one hot encoding ---
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

### --- sequential modeling ---
model = Sequential()

### --- building the neural network ---
number_inputs = 14
number_hidden_nodes = 7
model.add(Dense(units=number_hidden_nodes, activation='relu', input_dim=number_inputs))

number_classes = 2
model.add(Dense(units=number_classes, activation='softmax'))

### --- check model architecture ---
print(model.summary())

### --- compile;fit;loss check ---
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train_scaled, y_train_categorical, epochs=1000, shuffle=True, verbose=2)

model_loss, model_accuracy = model.evaluate(X_test_scaled, y_test_categorical, verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
'''

## ---------------- Stop Time ----------------
end_time = time()
elapsed = end_time - start_time
print(f"Algorithm time: {elapsed} seconds")
print()
print()

