For this week, we will be working with data wrangling. Data wrangling is the process of cleaning, structuring and enriching raw data into a desired format for better decision making in less time. It is a time-consuming process and usually a big chunk of the time in a data science project.

Unless stated otherwise, you can choose to work with one of the following libraries: Pandas, Polars or DuckDB.
# Task 1: Aggregations

# Task 2: Joins

# Task 3: Window Functions

# Task 4: Combined methods

# Task 5: 3 dataset, 3 libraries

## Story:
Here is a dataset which needs to be wrangled. You should try using Pandas, Polars and Duckdb to wrangle the data. Which library do you think is the best for wrangling this dataset? Which library is the fastest? Which library do you think is the easiest to use?

## Tasks:
- Load the dataset using Pandas, Polars and Duckdb.

## Dataset:
```python
import pandas as pd
import polars as pl
import duckdb

# Read parquet file and csv files

...

# Display dataframe
```

## Expected Output:
```python
# Pandas
import pandas as pd

df = pd.read_parquet('final_results.parquet')
print(df)
```

# Task 6: Wrangle and explain

## Story: 
You are now working as a data analyst and your boss asks you what the relationship between the number of hours spent on the company's online platform and the revenue generated is. You have access to the data but you need to clean it up and interpret the results for your boss.

## Tasks:
- Load the dataset and inspect the first few rows.
- Check for missing values and handle them appropriately.
- Determine the relationship and show using a visualization.
- Make a powerpoint presentation to explain the relationship to the class next week.


