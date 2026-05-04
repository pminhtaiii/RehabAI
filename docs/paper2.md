# Exercise-Specific Feature Extraction Approach for Assessing Physical Rehabilitation[cite: 2]

**Authors:** Qingyang Guo¹, Shehroz S. Khan²[cite: 2]  
**Affiliations:**  
¹ Division of Engineering Science, Faculty of Applied Science and Engineering, University of Toronto[cite: 2]  
² KITE - Toronto Rehabilitation Institute, University Health Network, Canada[cite: 2]  
**Emails:** clara.guo@mail.utoronto.ca, shehroz.khan@uhn.ca[cite: 2]  

---

## Abstract
Remote monitoring of physical exercises in a telerehabilitation program is a challenging task due to several technological factors, including varying video quality, lighting conditions, occlusions and camera angles[cite: 2]. Traditional approaches include marker-based or Kinect-based systems[cite: 2]. However, in a real-world system, those facilities may not be available and people may use regular web cameras available on their devices[cite: 2]. In this paper, we present a method to directly capture body poses from RGB videos, then extract various features from them to assess the quality of exercises[cite: 2]. Since different exercises may focus on different body parts and may require different movements; therefore, we extract exercise-specific features and trained various standard machine learning models[cite: 2]. To test our approach, we performed extensive experiments on KiMoRe dataset, which provides RGB videos of 5 different types of rehabilitation exercises and the score annotations assigned by trained clinicians[cite: 2]. We show that the feature extraction based approach outperforms the baseline rule-based method in terms of superior Spearman's rank correlation coefficient[cite: 2].

---

## 1 Introduction
Participation in physical rehabilitation training is clinically proven to be effective in relieving pain and restoring functional abilities[cite: 2]. However, many patients with rehabilitation needs may have difficulties accessing clinical-based rehabilitation, due to distance, lack of transportation, and limited healthcare resources[cite: 2]. As a result, home-based rehabilitation programs are offered as an alternative[cite: 2]. Especially during the time of the COVID-19 pandemic, many rehabilitation centers have to reduce or suspend rehabilitation services in order to reduce the risk of spreading the virus[cite: 2]. Under such circumstances, rehabilitation sessions are delivered via mobile devices using video cameras and patients are tasked to perform the prescribed therapy exercises at home[cite: 2]. However, one major issue with the home exercise program is adherence, with estimates of non-adherence as high as 50%, caused by factors such as lack of monitoring and feedback, absence of real-time supervision, and low self-efficacy[cite: 2]. Poor adherence can largely compromise patient outcomes and prolong the treatment[cite: 2]. Thus, there is a demand for an automated system that can evaluate and provide immediate feedback on patient performance to support home-based rehabilitation[cite: 2].

---

## 2 Related Work
Many of the existing work on action quality assessment was done in the sports settings and the data was collected from healthy athletes[cite: 2]. However, in the context of physical rehabilitation, the participants are patients with different medical conditions and their postures when performing the exercises can vary a lot from normal actions[cite: 2]. Most of the existing approaches for automated assessment of patient performance in rehabilitation exercises, such as rule-based and template-based approaches, are lacking generalization and robustness[cite: 2]. Another impediment of the existing methods is their reliance on motion sensing devices for data collection, such as Microsoft Kinect sensor, which had been discontinued by Microsoft[cite: 2]. There are other similar depth cameras available in the market, e.g. Orbbec; however, getting technical support for them can be challenging[cite: 2]. Besides, it also means installation of an additional piece of hardware that requires cost and changes in living space[cite: 2]. To address these shortcomings in the existing methods, we aim to develop an automated system to assess the quality of different physical rehabilitation exercises by only using the RGB videos[cite: 2]. In this paper, we develop an exercise-specific feature extraction approach that outperforms rule-based method on a publicly available physical rehabilitation dataset[cite: 2]. 

The significance and the novelty of the proposed work is that it supplements the traditional approaches for exercise assessment by using machine learning to automate the process of evaluating action quality[cite: 2]. In addition, comparing to the deep learning-based approaches, the feature extraction based approach provides a higher level of interpretability, which is highly valued in the healthcare domain[cite: 2].

---

## 3 Dataset
To train and validate our models, we used the KiMoRe dataset[cite: 2]. To the best of our knowledge, it is the only existing publicly available dataset for physical rehabilitation exercises that provides both RGB video data and score annotations assigned by trained clinicians[cite: 2]. The KiMoRe dataset has 78 participants in total, including 44 healthy participants (15 females, age mean±std: 36.7±16.8) and 34 participants living with chronic motor disabilities (19 females, 60.44±14.2), including Parkinson's Disease (n=16), Cerebral Stroke (n=10), and back pain (n=8)[cite: 2]. 

The participants are tasked to perform 5 exercises (Ex1-Ex5) that are commonly used for physiotherapy of axial disorders, as demonstrated in Figure 1[cite: 2]. 
*   Ex1 involves lifting of arms[cite: 2]. The participants need to hold a bar with both hands and raise the arms above the head while keeping the elbows straight[cite: 2]. 
*   Ex2 involves lateral tilting of the torso[cite: 2]. The participants need to stretch both arms straight and raise them above the head, and then slowly tilt the torso first to the left then to the right[cite: 2]. 
*   Ex3 involves the rotation of the torso[cite: 2]. The participants need to stretch both arms forward and then rotate his/her torso first to the left then to the right[cite: 2]. To correctly perform Ex2 and Ex3, the participants need to keep the elbows straight and avoid bending the torso forward or backward during the exercise[cite: 2]. 
*   Ex4 involves pelvis rotation on the transverse plane[cite: 2]. The participants need to stand still with feet slightly apart and make a circular rotation with the pelvis first in a clockwise then in a counter-clockwise direction[cite: 2]. 
*   Ex5 is squatting and the participants need to keep both arms stretched forward during the exercise[cite: 2]. 

KiMoRe dataset provides three types of highly correlated scores: clinical Total Score (CTS), clinical Primary Outcome (CPO) score and clinical Control Factors (CCF)[cite: 2]. The CTS score is the sum of the CPO score and CCF score and its value ranges from 10 to 50[cite: 2]. A higher score indicates the exercise was performed more accurately[cite: 2]. The CTS scores are used as the ground truth for training all the models discussed in the Experiment section[cite: 2]. The dataset provides RGB videos, depth videos, and 3D body joints collected by Microsoft Kinect v2 sensor[cite: 2]. In this work, we only use the RGB videos captured at 25 fps with a resolution of 960x540[cite: 2]. In each video, the participant performs the exercise for five consecutive times[cite: 2]. However, the duration of these exercises may vary depending upon the speed at which the participants perform the exercises, and thus the video lengths are of variable duration[cite: 2].

---

## 4 Proposed Method
The framework to assess the quality of exercise consists of three main components: skeletal joints extraction, frame selection, and exercise-specific feature extraction[cite: 2]. The general architecture of the proposed framework is shown in Figure 2[cite: 2]. To begin the preprocessing, firstly the videos are split into frames[cite: 2]. Then, the 2D-coordinates of the skeletal joints are extracted from each frame of the RGB video using OpenPose library[cite: 2]. However, OpenPose may sometimes extract erroneous body points and label them as zero, thus making the 2D joints irrelevant for analysis[cite: 2]. This could happen if some parts of the person's body is out of view of the camera or due to poor lighting conditions or far away distance from the camera[cite: 2]. The frames containing these erroneous body joint values are termed as 'invalid frames'[cite: 2]. To address this, a frame selection algorithm is applied to filter out the invalid frames and select the significant frames which contain the 'peak' postures[cite: 2]. Using the significant frames, the exercise-specific features are extracted from the skeletal data[cite: 2]. Finally, the features are concatenated from each significant frames and passed to different regression models to assess the quality of exercises[cite: 2].

### 4.1 Skeletal Joints Extraction
To extract the 2D-coordinates of skeletal joints from the RGB videos, each video is split into frames[cite: 2]. OpenPose BODY_25 model is run on each frame and generated a JSON file containing the x- and y-coordinates of 25 body keypoints, as shown in Figure 3[cite: 2]. Using the generated JSON files, all the skeletal data extracted from a video can be put into an array with dimension of number_of_frames x 50 (each of the 25 body joints has x- and y-coordinate)[cite: 2].

### 4.2 Frame Selection
As discussed earlier, due to technical or camera related issues, all the body joints may not be extracted correctly by OpenPose, which replaces such erroneous points with zero values[cite: 2]. Therefore, among all the extracted frames from the original RGB videos, we extract a certain number of significant frames by eliminating the invalid frames and use them for feature extraction[cite: 2]. 

The validity of a frame is determined by whether the predefined essential keypoints of that type of exercise can be correctly extracted by OpenPose from the selected frame[cite: 2]. All invalid frames are removed from further processing[cite: 2]. During the skeletal joints extraction process, it was found that the face keypoints 0, 15, 16, 17, 18 are missing from most of the frames, due to the blurring of the participants' faces, as shown in Figure 3[cite: 2]. In addition, several videos failed to capture the lower body parts of the participants, causing the Body 25 model fail to detect the lower-body keypoints {10, 11, 13, 14, 19, 20, 21, 22, 23, 24} from those videos[cite: 2]. To handle the issue with missing body joints, the face keypoints and the lower-body keypoints are excluded from the list of essential keypoints of Ex1, Ex2, Ex3, since these exercises are focusing on upper body movements[cite: 2]. Whereas, for Ex4 and Ex5, which require the movements of lower body parts, additional lower-body keypoints are added to the list of essential keypoints[cite: 2]. In summary, the essential keypoints needed for Ex1, Ex2, Ex3 are {1, 2, 3, 4, 5, 6, 7, 8, 9, 12}, for Ex4 are {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13}, for Ex5 are {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}[cite: 2]. 

In the second stage of the frame selection process, the 'peak' posture frames are chosen among the valid frames[cite: 2]. The 'peak' postures are the postures that require the most effort to complete[cite: 2]. For example, for Ex1 (Lifting of the arms), the posture that requires the most effort is the one that requires participants to raise both arms above the head to reach for the highest point[cite: 2]. The peak postures of Ex1 to Ex5 are shown in Figure 4[cite: 2]. 

In each video, the participant repeats the same exercise 5 times[cite: 2]. Therefore, 5x number of peak postures frames can be extracted per video, where x is the number of 'peak' postures per repetition for a specific type of exercise[cite: 2]. Ex1 has x=1, resulting in 5 peak posture frames to be extracted per video[cite: 2]. For Ex2, Ex4, and Ex5, x=2, thus 10 peak posture frames can be extracted per video[cite: 2]. For Ex3, x=3, 15 peak posture frames can be extracted[cite: 2].

### 4.3 Feature Extraction
For each type of exercise, a set of predefined features are extracted from the peak posture frames[cite: 2]. The exercise-specific features for Ex1 to Ex5 are shown in Figure 5 and Table 1 shows a summary of the exercise-specific features[cite: 2].

*   **left_elbow_angle**: left elbow extension angle at joint 3[cite: 2]
*   **right_elbow_angle**: right elbow extension angle at joint 6[cite: 2]
*   **hand_shoulder_ratio**: hands distance to shoulder width ratio (d1/d2)[cite: 2]
*   **torso_tilted_angle**: angled formed by torso vector (joint 1 and 8) and a vertical vector[cite: 2]
*   **hand_tilted_angle**: angle formed by hand-to-hand vector and a horizontal vector[cite: 2]
*   **elbow_angles_diff**: absolute difference between left_elbow_angle and right_elbow_angle[cite: 2]
*   **left_shoulder_angle**: left shoulder angle elevation at joint 2[cite: 2]
*   **right_shoulder_angle**: right shoulder elevation angle at joint 5[cite: 2]
*   **left_arm_torso_angle**: angle formed between torso vector and left arm[cite: 2]
*   **right_arm_torso_angle**: angle formed between torso vector and right arm[cite: 2]
*   **knee_hip_ratio**: knee distance to hip width ratio[cite: 2]
*   **shoulder_level_angle**: angle formed by shoulder_to_shoulder vector and a horizontal vector[cite: 2]
*   **left_knee_angle**: left knee flexion angle at joint 10[cite: 2]
*   **right_knee_angle**: right knee flexion angle at joint 13[cite: 2]

**Table 1: Exercise-specific features extracted for Ex1 to Ex5.**[cite: 2]

| ID | Feature | Ex1 | Ex2 | Ex3 | Ex4 | Ex5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | left_elbow_angle | | | | | ✓ |
| 1 | right_elbow_angle | | ✓ | ✓ | | ✓ |
| 2 | hand_shoulder_ratio | ✓ | ✓ | ✓ | | ✓ |
| 3 | torso_tilted_angle | ✓ | ✓ | ✓ | ✓ | ✓ |
| 4 | hand_tilted_angle | ✓ | | | | |
| 5 | elbow_angles_diff | | ✓ | ✓ | | |
| 6 | left_shoulder_angle | | ✓ | ✓ | | ✓ |
| 7 | right_shoulder_angle | | ✓ | ✓ | | ✓ |
| 8 | left_arm_torso_angle | | | ✓ | | |
| 9 | right_arm_torso_angle | | | ✓ | | |
| 10 | knee_hip_ratio | | | | ✓ | |
| 11 | shoulder_level_angle | | | | ✓ | |
| 12 | left_knee_angle | | | | | ✓ |
| 13 | right_knee_angle | | | | | |

---

## 5 Experiments and Results
To train and test the selected machine learning models, the extracted exercise-specific features are flattened to a 1D-array, which has the dimension of [number of peak posture frames * number of extracted features][cite: 2]. 

To select the best performed models, we applied 5-fold cross-validation (CV) to all the models using different features and computed the CV scores[cite: 2]. To be consistent with the existing literature, the Spearman's rank correlation coefficient is used as the metric (higher is better) to evaluate model performance in predicting the clinical scores[cite: 2]. The CV score is computed by taking the average of Spearman's rank correlation coefficient computed at each CV iteration[cite: 2]. The model and the features used to achieve the highest CV score for each type of exercise are reported in Table 2[cite: 2]. 

**Table 2: The best performed models for Ex1 to Ex5 along with the features used to achieve the highest CV scores.**[cite: 2]

| Exercise | 5-fold CV Spearman's | Model | Features Used |
| :--- | :--- | :--- | :--- |
| Ex1 | 0.55 | SVM | 0, 1, 2, 3, 4, 5 |
| Ex2 | 0.64 | RF | 0, 1, 3, 5, 6, 7 |
| Ex3 | 0.63 | RF | 0, 1, 2, 3, 5, 6, 7, 8, 9 |
| Ex4 | 0.37 | KNN-5 | 3, 10 |
| Ex5 | 0.42 | RF | 0, 1, 2, 3, 5, 6, 7 |

Furthermore, the best performed models are compared with the baseline rule-based model and the comparative results are reported in Table 3[cite: 2].

**Table 3: The comparative results of feature extraction based approach and the baseline. The average Spearman's rank correlation coefficients generated from 5-fold cross validation are reported.**[cite: 2]

| Method | Ex1 | Ex2 | Ex3 | Ex4 | Ex5 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Feature-Based | 0.55 | 0.64 | 0.63 | 0.37 | 0.42 |
| Rule-Based (baseline) | 0.44 | 0.41 | 0.46 | 0.62 | 0.30 |

As shown in Table 3, the feature extraction based approach outperforms the baseline rule-based method for all types of exercises, except for Ex4[cite: 2]. One possible explanation is that the proposed features are good representations for 2D motion (moving left/right, up/down), but not for 3D motion[cite: 2]. Ex4 involves circular motion but depth information is absent in 2D, therefore the extracted features might not be sufficient for the model to learn the relationship between body movements and the clinical score of this type of exercise[cite: 2].

---

## 6 Conclusions
Due to the variability in different exercises in a remote physical rehabilitation program, we proposed an exercise-specific feature extraction approach for assessing the performance of physical rehabilitation exercises using only RGB videos[cite: 2]. The results on a publicly available dataset show that our approach outperforms the baseline for all types of exercises, except for one[cite: 2]. In future, we will apply temporal relationships between exercise specific features using long short-term memory models[cite: 2]. In addition, we will investigate the effectiveness of using all angles for different exercises[cite: 2]. We will also develop sex/age-specific models to understand their impact on the performance[cite: 2].