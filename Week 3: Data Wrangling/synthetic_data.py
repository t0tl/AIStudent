# /// script
# dependencies = [
#   "duckdb==1.1.3",
#   "pandas==2.2.3",
#   "numpy==2.1.3",
#   "tqdm==4.67.1"
# ]
# ///
import uuid

import duckdb
import pandas as pd
import numpy as np
from tqdm import tqdm

# 1. `streaming_data` - Contains information about the streaming sessions, including `title_id`, `bandwidth`, `time_measured`, `region`, `resolution`, and `device`.

# 2. `title_data` - Contains details about the titles, including `title_id`, `title_name`, `genre`, and `release_year`.

# 3. `viewership_data` - Contains data about the viewership of titles, including `title_id`, `viewership`, and `time_day`.


# Create a database called BingeBlitz.db
con = duckdb.connect('BingeBlitz.db')

# Add noise to bandwidth amounts
BANDWIDTH_AMOUNTS = [100, 150, 200, 1000]
REGIONS = ['US', 'UK', "SE", "JP"]
RESOLUTIONS = ['720p', '1080p', '4K']
DEVICES = ['Samsung TV', 'Apple TV', 'Roku', 'Fire TV', 'Chromecast', 'Xbox', 'Playstation', 'Windows', 'Mac', 'iPhone', 'Android']
# Add offset to the time measured
TIME_MEASURED = []
for i in range(24):
    for j in range(60):
        TIME_MEASURED.append(f"2021-01-01 {i:02d}:{j:02d}:{np.random.randint(0, 59):02d}")

# Read movies.csv to get title name, genre and release year
movie_df = con.sql(
    """
    SELECT Title, Year, Genre FROM read_csv_auto('movies.csv');
    """
).to_df()

movie_df.columns = ['title', 'year', 'genre']

# Remove the first genre to get some null values
movie_df['genre'] = movie_df['genre'].str.split(", ").apply(lambda x: x[1:] if len(x) > 1 else None)
movie_df['title_id'] = [uuid.uuid4() for _ in range(len(movie_df))]
title_id_df = movie_df['title_id'].copy()

con.sql("DROP TABLE IF EXISTS title_data;")

con.sql(
    """
    CREATE TABLE title_data AS (
        SELECT title, year, genre, title_id FROM movie_df
    )
    """
)
print("Created title_data table.")
print("Creating viewership table...")


TIME_DAY = []
for j in range(1, 31):
    for k in range(24):
        TIME_DAY.append(f"2021-01-{j:02d} {k:02d}:00:00")

viewership_df = pd.DataFrame({
    'title_id': np.random.choice(title_id_df, 100),
    'viewership': np.random.randint(100, 100_000, 100),
    'time_day': TIME_DAY[0]
})

con.sql("DROP TABLE IF EXISTS viewership_data;")

con.sql(
    """
    CREATE TABLE viewership_data AS (
        SELECT * FROM viewership_df
    );
    """
)

for time_day in tqdm(TIME_DAY[1:]):
    viewership_df = pd.DataFrame({
            'title_id': np.random.choice(title_id_df, 100),
            'viewership': np.random.randint(100, 100_000, 100),
            'time_day': np.array(time_day).repeat(100)
        })
    con.sql(
        """
        INSERT INTO viewership_data
        SELECT * FROM viewership_df
        """
    )

print("Created viewership table.")
print("Creating streaming_data table...")

N_ROWS_STREAMING = 50_000_000

N_BATCHES = 1000

N_ROWS_BATCH = N_ROWS_STREAMING // N_BATCHES

streaming_data_df = pd.DataFrame({
        'title_id': np.random.choice(title_id_df, N_ROWS_BATCH),
        # use float16 to save space
        'bandwidth': np.random.normal(
            np.random.choice(BANDWIDTH_AMOUNTS, N_ROWS_BATCH),
            25
        ).astype('float16'),
        'time_measured': np.random.choice(TIME_MEASURED, N_ROWS_BATCH),
        'region': np.random.choice(REGIONS, N_ROWS_BATCH),
        'resolution': np.random.choice(RESOLUTIONS, N_ROWS_BATCH),
        'device': np.random.choice(DEVICES, N_ROWS_BATCH)
    })

con.sql("DROP TABLE IF EXISTS streaming_data;")

con.sql(
    """
    CREATE TABLE streaming_data AS (
        SELECT
            title_id,
            bandwidth,
            time_measured::TIMESTAMP AS time_measured,
            region,
            resolution,
            device
        FROM streaming_data_df
    );
    """
)

for batch in tqdm(range(N_BATCHES-1)):
    streaming_data_df = pd.DataFrame({
        'title_id': np.random.choice(title_id_df, N_ROWS_BATCH),
        # use float16 to save space
        'bandwidth': np.random.normal(
            np.random.choice(BANDWIDTH_AMOUNTS, N_ROWS_BATCH),
            25
        ).astype('float16'),
        'time_measured': np.random.choice(TIME_MEASURED, N_ROWS_BATCH),
        'region': np.random.choice(REGIONS, N_ROWS_BATCH),
        'resolution': np.random.choice(RESOLUTIONS, N_ROWS_BATCH),
        'device': np.random.choice(DEVICES, N_ROWS_BATCH)
    })
    con.sql(
        """
        INSERT INTO streaming_data
        SELECT
            title_id,
            bandwidth,
            time_measured::TIMESTAMP AS time_measured,
            region,
            resolution,
            device
        FROM streaming_data_df
        """
    )

print("Created streaming_data table.")

con.close()