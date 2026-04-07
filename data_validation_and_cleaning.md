# Skill: Data Validation and Cleaning

## Description
Detects and corrects data quality issues including missing values, invalid entries, and inconsistencies.
Categorical errors → rows removed  
Numerical errors → imputed using median, mean, or mode.

## Prompt
Clean the dataset:
- Remove rows with invalid/missing categorical values
- Replace invalid numerical values (e.g., negative Age) with NaN
- Impute:
  - Median → skewed features
  - Mean → symmetric features
  - Mode → discrete values
- Ensure no missing numerical values remain
