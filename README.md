# Multi-Model System for Trip Description, Location Classification, Rating Prediction, and Intent Chatbot

## Project Overview

This repository contains a set of models designed for various tasks related to trip data. The models include:

1. **Trip Description Generation**: A sequence-to-sequence model that generates trip descriptions based on location and type.
2. **Location Classification**: A classification model that predicts location categories (Sharjah City, East Coast, and Central Region) based on trip descriptions.
3. **Rating Prediction**: A regression model that predicts user ratings based on trip location, activity, and budget.
4. **Intent Chatbot**: A chatbot model that recognizes user intents and responds with appropriate activities and currency information based on Sharjah City.

## Repository Structure

```bash
├── location_classification/
│   ├── models/
│   │   ├── DTclassification.joblib
│   │   └── TFIDFDTclassification.joblib
│   ├── notebooks/
│   │   └── Classification-location.ipynb
│   ├── scripts/
│   │   └── locationclassifier.py
│
├── trip_description/
│   ├── models/
│   │   ├── T5-Generation-Description-Tokenizer
│   │   └── T5-Generation-Description-Model-Two.rar
│   ├── notebooks/
│   │   ├── T5-base-Description.ipynb
│   │   ├── T5Description-small.ipynb
│   │   └── gpt2description.ipynb
│   ├── scripts/
│   │   └── tripgeneration.py
│
├── trip_rating/
│   ├── models/
│   │   ├── RF-Regrassion-Rating.joblib
│   │   └── TFIDFDTregression.joblib
│   ├── notebooks/
│   │   └── Rating-Regression.ipynb
│   ├── scripts/
│   │   └── ratingregressor.py
│
├── intents_chatbot/
│   ├── models/
│   │   └── intents-chatbot.keras
│   ├── notebooks/
│   │   └── intents-chatbot.ipynb
│   ├── scripts/
│   │   └── IntentsChatbot.py
│
├── LICENSE
└── README.md
```
## Models

### 1. **Trip Description Generation (T5-based)**

- **Task**: Generates a detailed trip description based on input location and type.
- **Model**: T5-small (pretrained)
- **Metrics**: 
  - BLEU: 0.71
  - ROUGE-1: 0.81
  - ROUGE-2: 0.76
  - ROUGE-L: 0.81

### 2. **Location Classification**

- **Task**: Classifies the location into one of three categories: Sharjah City, East Coast, and Central Region.
- **Model**: Decision Tree Classifier
- **Metrics**:
  - Accuracy: 97%
  - Precision (Sharjah City): 99%
  - Recall (East Coast): 99%
  - F1-Score (Central Region): 99%

### 3. **Rating Prediction**

- **Task**: Predicts ratings for a trip based on the location, activity, and budget (low/medium/high).
- **Model**: Random Forest Regressor
- **Metrics**:
  - R²: 0.92
  - Mean Squared Error (MSE): 0.05
  - Mean Absolute Error (MAE): 0.04
  - Explained Variance Score: 0.92

### 4. **Intent Chatbot (BERT-based)**

- **Task**: Identifies the user’s intent related to activities in Sharjah City.
- **Model**: BERT-based architecture for intent classification
- **Metrics**:
  - Accuracy: 99.45%
  - Loss: 0.0485

## How to Use

### Prerequisites

- Python 3.x
- Required Python Libraries: tensorflow, transformers, scikit-learn, nltk, pandas, rouge_score, keras

### Step 1: Clone the repository
```bash
git clone https://github.com/MinaIbrahim10/Trip-generation.git
cd Trip-generation
 ```

Step 2: Install the required libraries and Make sure you have Python installed
 ```bash
python --version
 
pip install -r requirements.txt
   ```
Step 3: Run the models
For example, to use the Location Classification model, run:
 ```bash
python location_classification/scripts/locationclassifier.py
```
Step 3: Follow the on-screen prompts to enter the trip details:
    - description (string)
The script will output a predicted  trip location.
## Model Performance

    Trip Description Generation (T5): Achieved a BLEU score of 0.71 and high ROUGE metrics (ROUGE-1: 0.81, ROUGE-2: 0.76,ROUGE-L: 0.81).
    Location Classification (Decision Tree): Achieved an accuracy of 97% on the test data.
    Rating Prediction (Random Forest): Achieved an R² score of 0.92 and excellent regression metrics.
    Intent Chatbot (BERT): Achieved an accuracy of 99.45% in intent classification with low loss.
## Contributors

- **Mina Ibrahim**: Developer, model training, and deployment.
- **Karam Ghazy** ([karam-ghazy](https://github.com/karam-ghazy)): Developer, model training, and deployment, with support in data preprocessing and feature engineering.


## Future Work

- Fine-tuning the model with more data.
- Adding cross-validation and further performance metrics.
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

