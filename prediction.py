# -----------Time------------
from time import time
start_time = time() 

# Dependencies
import pandas as pd 
import numpy as np

# Turn-off warnings
pd.options.mode.chained_assignment = None ## SettingWithCopyWarning

all_matches = pd.read_csv("results.csv")

# print(all_matches.head())
reduced_matches = all_matches[0-200:]
# print(reduced_matches.columns)

def winners(df):
	da_winners = []
	winners_score = []
	score_difference = []
	penalties = []
	for index, row in df.iterrows():
		if row['home_score'] > row['away_score']:
			da_winners.append(row['home_team'])
			winners_score.append(row['home_score'])
		else:
			da_winners.append(row['away_team'])
			winners_score.append(row['away_score'])

		score_difference.append(abs(row['home_score'] - row['away_score']))
		if abs(row['home_score'] - row['away_score']) == 0:
			penalties.append(True)
		else:
			penalties.append(False)

	df["winners"] = [winner for winner in da_winners]
	df["winners_score"] = [ws for ws in winners_score]
	df["score_difference"] = [diff for diff in score_difference]
	df["penalties"] = [p for p in penalties]

	return df

world_cup_winners = pd.read_csv("World Cup_Winners.csv")
print(world_cup_winners["Attendance"].dtype())

# winners(reduced_matches)
# print(reduced_matches.columns)

# winners(all_matches)
# print(all_matches.head())


# -----------Time------------
end_time = time()
elapsed = end_time - start_time
print(f"Algorithm time: {elapsed} seconds")




