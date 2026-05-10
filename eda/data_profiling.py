
import pandas as pd


data = pd.read_csv('/content/drive/MyDrive/rehab-ai-data/KiMoRe_final/KiMoRe_data_blur.csv').drop('Unnamed: 0', axis=1)
def profile(exercise, ptype):
    """Print profiling statistics for an exercise and participant type."""
    dataEs = data[(data['exercise'] == exercise) & data['ID'].str.startswith(ptype)]
    
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


profile('Es1', 'E')
profile('Es1', 'NE')
profile('Es1', 'P')
profile('Es1', 'B')
profile('Es1', 'S')

profile('Es2', 'E')
profile('Es2', 'NE')
profile('Es2', 'P')
profile('Es2', 'B')
profile('Es2', 'S')
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
