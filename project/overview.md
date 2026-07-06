Overview
🍽️ Can your model eat with its eyes?

Estimating the calorie content of food from a single photograph is one of the most practical and challenging problems in applied computer vision. Unlike controlled recognition tasks, real-world food images vary in camera angle, lighting, plate presentation, portion size, and dish complexity — making generalization the central difficulty.

In this challenge you are given a dataset of real food dish images drawn from two distinct sources with verified calorie labels. Your goal is to train a deep learning model that predicts the total calorie content of a dish from a single RGB image.

What makes this hard?

Calories cannot be read from appearance alone. A bowl of pasta and a bowl of salad may look similar in size but differ by 600 kcal. Your model must learn visual representations that capture not just what food is present, but implicitly reason about ingredient density and portion — all from pixels.

The training and test images come from two sources with different camera styles, lighting conditions, and plating conventions. A model that overfits to one visual style will generalize poorly. A model that learns robust food representations will not.

Description

The dataset combines two real-world food image sources, each with verified ground-truth calorie annotations. Source identities are not disclosed, this is intentional 😉. The two sources differ in camera style, angle, and lighting. Generalization across these visual domains is part of the challenge.

Files

train/images/ — 3,098 RGB food dish images for training

train_labels.csv — calorie label per training image image_id, filename, calories train_0000, train_0000.png, 193.0 train_0001, train_0001.jpg, 95.7

test/images/ — 547 RGB food images, no labels provided

test_ids.csv — image IDs for the test set image_id, filename test_0000, test_0000.png

sample_submission.csv — required submission format

Important notes

Images are not resized — handle resizing inside your DataLoader
Calorie values are continuous floats — this is a regression task
The calorie distribution is right-skewed — you should maybe consider log-scale prediction for training stability
No segmentation masks are provided
Source information has been removed from all filenames and image metadata intentionally
Rules

All training must run inside a Kaggle notebook — no external GPU
Maximum 10 submissions per day
Your notebook must be fully reproducible — we will re-run it.
Keep your notebooks private during the competition. Make them public only after the submission deadline has passed.
No external datasets are allowed
Final submission must be made at least 48 hours before oral defense

Evaluation

Submissions are scored on Mean Absolute Error (MAE):

MAE = mean( |predicted_calories − true_calories| )
content_copy
Lower is better.

Scoring

The test set is split into a public subset (30%) and a private subset (70%).

During the competition: your score is computed on the public subset only
After the deadline: final rankings use the private subset score
This means your public leaderboard position may differ from your final grade ranking. Do not overfit to the public score.

Submission File

Your submission must be a CSV file with exactly two columns:

image_id, predicted_calories
image_id must match the values in test_ids.csv exactly
predicted_calories is your model's calorie estimate in kcal
The file must contain exactly 547 rows (one per test image)
Do not include a header other than the column names above
Example

image_id, predicted_calories
test_0000, 450.0
test_0001, 320.5
test_0002, 189.0
...