# FIFA Data Analytics

## Project Overview

This project focuses on analyzing real-life player data and FIFA ratings using data analytics techniques. The main goals include:

1. Converting provided Excel files into .csv files for easier data analysis.
2. Using scatter plots to identify potential issues in the data.
3. Exploring both supervised and unsupervised learning methods.
4. Applying supervised learning models to our labeled dataset.
5. Evaluating the performance of different models, specifically decision trees and random forests.
6. Assessing the accuracy of the models in matching FIFA data with real-life player data.

## Repository Structure

- `data/` : Contains the original Excel files and the converted .csv files.
  - `Data.xlsx`
  - `modified_data.csv`
  - `modified_df_DEF.csv`
  - `modified_df_MID.csv`
  - `modified_df_OFF.csv`
  - `modified_df_data.csv`
- `notebooks/` : Jupyter notebooks used for data analysis and model training.
  - `gab_randomForest copy.ipynb`
- `plots/` : Generated scatter plots and other visualizations.
  - `Plot_DEF_Clearances/`
  - `Plot_Goals.py`
  - `Plot_Man_of_the_match.py`
  - `Plot_Rating.py`
- `scripts/` : Python scripts for data analysis and model implementation.
  - `Analysis_DATA_DT.py`
  - `Analysis_DATA_RF.py`
  - `Analysis_DEF_DT.py`
  - `Analysis_DEF_RF.py`
  - `Analysis_MID_DT.py`
  - `Analysis_MID_RF.py`
  - `Analysis_OFF_DT.py`
  - `Analysis_OFF_RF.py`
  - `DA_DecisionTree_OFF.py`
  - `DA_FIFA.py`
  - `DA_SplittingData.py`
- `documentation/` : Project documentation.
  - `doc_anish_linda/`
  - `docu_Gab/`
- `README.md` : Project overview and instructions.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.8+
- Jupyter Notebook
- Required Python libraries (listed in `requirements.txt`)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/FIFA_Data_Analytics.git
   cd FIFA_Data_Analytics
   ```

2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation

1. Convert the provided Excel files into .csv files:
   - Use the script `scripts/DA_SplittingData.py` to automate this process.

### Data Analysis

1. Use scatter plots to visualize the data:
   - Notebooks in the `notebooks/` directory demonstrate how to create scatter plots to identify issues within the data.

### Machine Learning Models

#### Supervised Learning

1. Decision Trees:
   - Implemented but did not yield satisfactory results.
   - Refer to `scripts/Analysis_DATA_DT.py` and `scripts/DA_DecisionTree_OFF.py` for details.

2. Random Forests:
   - Provided better results with a higher R-squared value of 0.8.
   - Refer to `notebooks/gab_randomForest copy.ipynb` and `scripts/Analysis_DATA_RF.py` for the analysis and results.

## Results

- Our analysis indicated that the models achieved up to 30% accuracy in matching FIFA ratings with real-life player data.
- We therefore realized that a decision tree is not the best option for our cause.
- The random forest gave us a significantly higher R^2, especially when we focused on the important attributes of the respective tables.

## Team Members

- Linda Blumenthal
- Deema Aassy
- I-En Hung
- Marco Schneider
- Anish Biswas

## Conclusion

Despite extensive analysis and the application of various supervised learning models, we found that the accuracy of matching FIFA data with real-life player data was relatively low. Future work could explore additional models, feature engineering, or data augmentation techniques to improve performance.

## Contact

For any questions or contributions, please open an issue or contact one of the team members.

---

We hope this project serves as a useful resource for understanding the application of data analytics and machine learning in the context of sports data. Thank you for your interest!

