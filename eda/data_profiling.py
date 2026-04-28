"""
Auto-converted from data_profiling.ipynb on 2026-04-25 19:04:47
"""

# %%
# Importing necessary packages
import pandas as pd

# %%
# Reading the CSV file into a DataFrame and dropping the 'Unnamed: 0' column
data = pd.read_csv('/content/drive/MyDrive/rehab-ai-data/KiMoRe_final/KiMoRe_data_blur.csv').drop('Unnamed: 0', axis=1)
data

# %%
# Function for Data Profiling Overview
def profile(exercise,ptype):
  # Filtering data based on exercise and participant type
  dataEs = data[(data['exercise'] == exercise) & data['ID'].str.startswith(ptype)]
  
  # Calculating various statistics
  participants = dataEs.shape[0]
  missing = dataEs[pd.isna(dataEs['video'])].shape[0]
  records = dataEs[pd.notna(dataEs['video'])]
  num_records = records.shape[0]
  max_clinical_score = records['clinical_score'].max()
  mean_clinical_score = records['clinical_score'].mean()
  min_clinical_score = records['clinical_score'].min()
  max_frames = records['#frames'].max()
  mean_frames = records['#frames'].mean()
  min_frames = records['#frames'].min()

  # Printing the results
  print(f'\nExercise: {exercise} - Category: {ptype}\n'
        f'\t#Participants: {participants}\n'
        f'\t#Missing: {missing}\n'
        f'\t#Available Records: {num_records}\n'
        f'\t#Minimum Clinical Score: {min_clinical_score}\n'
        f'\t#Mean Clinical Score: {mean_clinical_score}\n'
        f'\t#Maximum Clinical Score: {max_clinical_score}\n'
        f'\t#Minimum #frames: {min_frames}\n'
        f'\t#Mean #frames: {mean_frames}\n'
        f'\t#Maximum #frames: {max_frames}')

# %%
# Profiling for Exercise 1 and participant type combinations
profile('Es1', 'E')
profile('Es1', 'NE')
profile('Es1', 'P')
profile('Es1', 'B')
profile('Es1', 'S')

# %%
# Profiling for Exercise 2 and participant type combinations
profile('Es2', 'E')
profile('Es2', 'NE')
profile('Es2', 'P')
profile('Es2', 'B')
profile('Es2', 'S')

# %%
# Profiling for Exercise 3 and participant type combinations
profile('Es3', 'E')
profile('Es3', 'NE')
profile('Es3', 'P')
profile('Es3', 'B')
profile('Es3', 'S')

# %%
# Profiling for Exercise 4 and participant type combinations
profile('Es4', 'E')
profile('Es4', 'NE')
profile('Es4', 'P')
profile('Es4', 'B')
profile('Es4', 'S')

# %%
# Profiling for Exercise 5 and participant type combinations
profile('Es5', 'E')
profile('Es5', 'NE')
profile('Es5', 'P')
profile('Es5', 'B')
profile('Es5', 'S')

# %%
