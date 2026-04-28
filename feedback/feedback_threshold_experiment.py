"""
Auto-converted from feedback_threshold_experiment.ipynb on 2026-04-25 19:04:47
"""

# %%
#Installing necessary packages
# Notebook-only setup command kept as comment for reference:
# !pip install -q tslearn

# %%
#Importing necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tslearn.metrics import dtw
from random import randrange
from IPython.display import HTML
from base64 import b64encode

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
# Reading the CSV data file into a Pandas DataFrame and dropping the 'Unnamed: 0' column
data = pd .read_csv('/content/drive/MyDrive/rehab-ai-data/KiMoRe_final/KiMoRe_data_movenet.csv').drop('Unnamed: 0', axis=1)

# %%
# Function to display a video given its path using HTML and base64 encoding
def display_video(path):
  # Read the video file and encode it in base64
  mp4 = open(path,'rb').read()
  data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
  # Return HTML code to display the video
  return HTML("""
  <video width=400 controls>
        <source src="%s" type="video/mp4">
  </video>
  """ % data_url)

# %% [markdown]
# Chosen exercise for this experiment to find the threshold is Es2.
#
# The most important joints in this exercise are the shoulders, elbows and wrists.
#
# The chosen ground truth video is E_ID12 as it has the highest clinical score among all the experts with 50.

# %%
data[(data['exercise'] == 'Es2') & data['ID'].str.startswith('E')].sort_values(by='clinical_score', ascending=False)

# %%
# Displaying a video from the specified path
display_video('/content/drive/MyDrive/rehab-ai-data/KiMoRe_rgb_movenet/CG/Expert/E_ID12/Es2/rgb/Blur_rgb011214_105527.mp4')

# %%
# Reading reference data from a CSV file
reference = pd.read_csv('/content/drive/MyDrive/rehab-ai-data/KiMoRe_rgb_movenet/CG/Expert/E_ID12/Es2/E_ID12_Es2.csv')

# %% [markdown]
# # DTW between Ground Truth (GT) and the rest of the subjects

# %%
# Listing of body joints for DTW calculation
body_joints = [
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
]

# %%
# Function to calculate DTW values for a specific exercise type and return a dictionary
def dtwValues(stype):
  # Filtering subjects based on exercise type and extracting joint positions data
  subjects = data[(data['exercise'] == 'Es3') & data['ID'].str.startswith(stype)]['joint_positions']
  dict = {}

  for subject in subjects:
    # Skipping NaN values
    if type(subject) == float:
      continue
    id = subject.split('/')[8]
    s = pd.read_csv(subject)

    # Skipping the reference ID
    if id == 'E_ID12':
      continue

    # Initializing a list for each subject in the dictionary
    dict[id] = []

    # Calculating DTW values for each body joint and appending to the list
    for bj in body_joints:
      joints = [f'{bj}_x', f'{bj}_y']
      for j in joints:
        dtw_value = dtw(reference[j], s[j])
        dict[id].append(dtw_value)

  return dict

# %%
def plot_distribution(subjectsDTW, stype):
  # Extracting keys (IDs) and values (DTW lists) from the dictionary
  keys = list(subjectsDTW.keys())
  values = list(subjectsDTW.values())

  fig, axs = plt.subplots(2, 1, figsize=(8, 8))
  fig_title = f'DTW Distribution for the {stype} Participants'
  plt.suptitle(fig_title, fontsize=15)

  #Bar Plot
  plt.subplot(2, 1, 1)
  means = [np.mean(val) for val in values]
  sns.barplot(y=keys, x=means, capsize=5, orient='h', color='#0E899B')
  plt.xticks(np.arange(plt.xlim()[0], plt.xlim()[1]+0.5, 0.5))

  plt.xlabel('Mean DTW Values')
  plt.ylabel('Participant ID')
  plt.title(f'DTW Plot of Mean Values for Each {stype} Participant')
  plt.grid(True)

  #BoxPlot
  plt.subplot(2, 1, 2)
  data = pd.DataFrame({key: value for key, value in zip(keys, values)})
  sns.boxplot(data=data, orient='h', color='#0E899B')
  plt.xlabel('DTW Values')
  plt.ylabel('Participant ID')
  plt.title(f'Boxplot for the DTW Values for Each {stype} Participant')
  plt.grid(True)

  # Adjusting layout, saving and displaying plots
  plt.tight_layout()
  plt.savefig(f'/content/drive/MyDrive/rehab-ai-data/feedback_threshold_experiment_plots/{fig_title}.png', dpi=300)
  plt.show()

  # Displaying overall min, mean, and max of mean DTW values
  print(f"Min : {np.min(means)} \t Mean : {np.mean(means)} \t Max : {np.max(means)}")

# %%
# Calculating DTW values for category type 'Experts' and plotting the distribution
dtw_experts = dtwValues('E')
plot_distribution(dtw_experts, 'Experts')

# %%
# Calculating DTW values for category type 'Non Experts' and plotting the distribution
dtw_nonexperts = dtwValues('NE')
plot_distribution(dtw_nonexperts, 'Non-Experts')

# %%
# Calculating DTW values for category type 'Parkinsons' and plotting the distribution
dtw_parks = dtwValues('P')
plot_distribution(dtw_parks, 'Parkinsons')

# %%
# Calculating DTW values for category type 'Back Pain' and plotting the distribution
dtw_backpain = dtwValues('B')
plot_distribution(dtw_backpain, 'Back Pain')

# %%
# Calculating DTW values for category type 'Stroke' and plotting the distribution
dtw_stroke = dtwValues('S')
plot_distribution(dtw_stroke, 'Stroke')

# %% [markdown]
# # DTW between GT and the rest of the subjects focusing only on the important joint for Exercise 2 with are the shoulders, elbow and wrist.

# %%
def dtwValues_focused(stype):
  # Filtering subjects based on exercise type and extracting joint positions data
  subjects = data[(data['exercise'] == 'Es3') & data['ID'].str.startswith(stype)]['joint_positions']
  dict = {}

  # Calculating DTW values for each important joint
  for subject in subjects:
    # Skipping NaN values
    if type(subject) == float:
      continue
    id = subject.split('/')[8]
    s = pd.read_csv(subject)

    # Skipping the reference ID
    if id == 'E_ID12':
      continue

    # Initializing a list for each subject in the dictionary
    dict[id] = []

    # Calculating DTW values for each important joint and appending to the list
    for bj in ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']:
      joints = [f'{bj}_x', f'{bj}_y']

      for j in joints:
        dtw_value = dtw(reference[j], s[j])
        dict[id].append(dtw_value)

  return dict

# %%
def plot_distribution_focused(subjectsDTW, stype):
  keys = list(subjectsDTW.keys())
  values = list(subjectsDTW.values())

  fig, axs = plt.subplots(2, 1, figsize=(8, 8))
  fig_title = f'DTW Distribution for the {stype} Participants Focused on Important Joints'
  plt.suptitle(fig_title, fontsize=15)

  #Bar Plot
  plt.subplot(2, 1, 1)
  means = [np.mean(val) for val in values]
  sns.barplot(y=keys, x=means, capsize=5, orient='h', color='#0E899B')
  plt.xticks(np.arange(plt.xlim()[0], plt.xlim()[1]+0.5, 0.5))

  plt.xlabel('Mean DTW Values')
  plt.ylabel('Participant ID')
  plt.title(f'DTW Plot of Mean Values for Each {stype} Participant')
  plt.grid(True)

  #BoxPlot
  plt.subplot(2, 1, 2)
  data = pd.DataFrame({key: value for key, value in zip(keys, values)})
  sns.boxplot(data=data, orient='h', color='#0E899B')
  plt.xlabel('DTW Values')
  plt.ylabel('Participant ID')
  plt.title(f'Boxplot for the DTW Values for Each {stype} Participant')
  plt.grid(True)

  # Adjusting layout, saving and displaying plots
  plt.tight_layout()
  plt.savefig(f'/content/drive/MyDrive/rehab-ai-data/feedback_threshold_experiment_plots/{fig_title}.png', dpi=300)
  plt.show()

  # Displaying overall min, mean, and max of max DTW values
  print(f"Min : {np.min(means)} \t Mean : {np.mean(means)} \t Max : {np.max(means)}")

# %%
# Calculating DTW values for 'Experts', focusing on important joints
dtw_experts = dtwValues_focused('E')
plot_distribution_focused(dtw_experts, 'Experts')

# %%
# Calculating DTW values for 'Non Experts', focusing on important joints
dtw_nonexperts = dtwValues_focused('NE')
plot_distribution_focused(dtw_nonexperts, 'Non-Experts')

# %%
# Calculating DTW values for 'Parkinsons', focusing on important joints
dtw_parks = dtwValues_focused('P')
plot_distribution_focused(dtw_parks, 'Parkinsons')

# %%
# Calculating DTW values for 'Back Pain', focusing on important joints
dtw_backpain = dtwValues_focused('B')
plot_distribution_focused(dtw_backpain, 'Back Pain')

# %%
# Calculating DTW values for 'Stroke', focusing on important joints
dtw_stroke = dtwValues_focused('S')
plot_distribution_focused(dtw_stroke, 'Stroke')

# %% [markdown]
# # Testing the possible candidates for the threshold
#
# ####  [1.5 - 2.0 - 2.5 - 3.0]

# %%
# Selecting stroke patients from the dataset for exercise 'Es2', sorting them based on 'joint_positions', and setting 'ID' as the index
stroke_patients = data[(data['exercise'] == 'Es2') & data['ID'].str.startswith('S')].sort_values(by='joint_positions').set_index('ID')
stroke_patients

# %%
# Defineing a function to provide feedback based on DTW values
def feedback(test,counter,reference,threshold):
  for bj in body_joints:
    joints = [f'{bj}_x', f'{bj}_y']
    for j in joints:
      dtwValue = dtw(test[j][:counter], reference[j][:counter])
      if dtwValue > threshold:
          axis = 'vertically' if j[-1:] == 'y' else 'horizontally'
          print(f'Adjust your {j.replace("_", " ")[:-2]} {axis}')

# %%
# Defining a function to test feedback for a specific ID
def test_feedback(id, test, reference, threshold):
  test_joints = pd.read_csv(test.loc[id,'joint_positions'])
  test_frames = test.loc[id,'#frames']

  print(f'The selected ID: {id}')

  for count in range(1,test_frames+1):
    if count % 30 == 0:
      print(f' {count} \t\t {feedback(test_joints, count, reference, threshold)}')

# %% [markdown]
# ## 1. First subject - S_ID7
# This stroke patient had a clinical score of 23.3
#
# It doesnot appear that this patient needs any feedback as they are performing the set exercise correctly as they possibly can considering their condition.

# %%
display_video('/content/drive/MyDrive/rehab-ai-data/KiMoRe_rgb_movenet/GPP/Stroke/S_ID7/Es2/rgb/Blur_rgb271114_112849.mp4')

# %%
test_feedback('S_ID7',stroke_patients,reference,1.5)

# %%
test_feedback('S_ID7',stroke_patients,reference,2)

# %%
test_feedback('S_ID7',stroke_patients,reference,2.5)

# %%
test_feedback('S_ID7',stroke_patients,reference,3)

# %% [markdown]
# ## 2. Second subject - S_ID8
#
# This stroke patient had a clinical score of 26.3
#
# This patient performed the exercise correctly but had to be corrected through out their performance by the specialists available with them.
#
# Our feedback model was able to capture those corrections.

# %%
display_video('/content/drive/MyDrive/rehab-ai-data/KiMoRe_rgb_movenet/GPP/Stroke/S_ID8/Es2/rgb/Blur_rgb271114_113727.mp4')

# %%
test_feedback('S_ID8',stroke_patients,reference,1.5)

# %%
test_feedback('S_ID8',stroke_patients,reference,2)

# %%
test_feedback('S_ID8',stroke_patients,reference,2.5)

# %%
test_feedback('S_ID8',stroke_patients,reference,3)

# %% [markdown]
# ## 3. Third Subject - S_ID1
#
# This stroke patient had a clinical score of 20.0
#
# This patient appeared to not be doing the exercise correctly from the beginning but this is due to their condition, however they are still being provided with the feedback to facilitate the healing process.

# %%
display_video('/content/drive/MyDrive/rehab-ai-data/KiMoRe_rgb_movenet/GPP/Stroke/S_ID1/Es2/rgb/Blur_rgb060616_122346.mp4')

# %%
test_feedback('S_ID1',stroke_patients,reference,1.5)

# %%
test_feedback('S_ID1',stroke_patients,reference,2)

# %%
test_feedback('S_ID1',stroke_patients,reference,2.5)

# %%
test_feedback('S_ID1',stroke_patients,reference,3)

# %% [markdown]
# # Chosen Threshold (T): 2.5
#
# This threshold was chosen due to the following reasons:
#
#
# 1.   T 1.5 was exaggerating the feedback
# 2.   T 2 was good but exaggerated the feedback as well and does not put into perspective that the patient's capabilities do not allow them to perform the exercise as accurately as the expert.
# 3.   T 3 was not capturing any corrections at all
#
# Therefore, T 2.5 was found to be a good balance.
#
# --------------------------------------------------------------------------------------------------------------------------------
#
# To further prove the capabilities of this threshold, a randomly selected id from the Parkinson's patients was selected and the threshold was tested on all the exercises.

# %%
# Randomly selecting an ID between 1 and 15
id = randrange(1,16)

# Filtering the DataFrame to get rows corresponding to the selected ID
final_test = data[data['ID'] == f'P_ID{id}'].set_index('ID')
final_test

# %%
#Es1
display_video(final_test[final_test['exercise'] == 'Es1'].loc[final_test.index[0], 'video'])

# %%
#Es1
gtEs1 = pd.read_csv('/content/drive/MyDrive/rehab-ai-data/KiMoRe_rgb_movenet/CG/Expert/E_ID15/Es1/E_ID15_Es1.csv')
test_feedback(final_test.index[0],final_test[final_test['exercise'] == 'Es1'],gtEs1,2.5)

# %%
#Es2
display_video(final_test[final_test['exercise'] == 'Es2'].loc[final_test.index[0], 'video'])

# %%
#Es2
gtEs2 = pd.read_csv('/content/drive/MyDrive/rehab-ai-data/KiMoRe_rgb_movenet/CG/Expert/E_ID12/Es2/E_ID12_Es2.csv')
test_feedback(final_test.index[0],final_test[final_test['exercise'] == 'Es2'],gtEs2,2.5)

# %%
#Es3
display_video(final_test[final_test['exercise'] == 'Es3'].loc[final_test.index[0], 'video'])

# %%
#Es3
gtEs3 = pd.read_csv('/content/drive/MyDrive/rehab-ai-data/KiMoRe_rgb_movenet/CG/Expert/E_ID12/Es3/E_ID12_Es3.csv')
test_feedback(final_test.index[0],final_test[final_test['exercise'] == 'Es3'],gtEs3,2.5)

# %%
#Es4
display_video(final_test[final_test['exercise'] == 'Es4'].loc[final_test.index[0], 'video'])

# %%
#Es4
gtEs4 = pd.read_csv('/content/drive/MyDrive/rehab-ai-data/KiMoRe_rgb_movenet/CG/Expert/E_ID1/Es4/E_ID1_Es4.csv')
test_feedback(final_test.index[0],final_test[final_test['exercise'] == 'Es4'],gtEs4,2.5)

# %%
#Es5
display_video(final_test[final_test['exercise'] == 'Es5'].loc[final_test.index[0], 'video'])

# %%
#Es5
gtEs5 = pd.read_csv('/content/drive/MyDrive/rehab-ai-data/KiMoRe_rgb_movenet/CG/Expert/E_ID1/Es5/E_ID1_Es5.csv')
test_feedback(final_test.index[0],final_test[final_test['exercise'] == 'Es5'],gtEs5,2.5)

# %%
