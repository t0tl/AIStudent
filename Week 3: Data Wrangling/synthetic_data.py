import duckdb
# from openai import OpenAI

# Create a database called BingeBlitz.db
con = duckdb.connect('BingeBlitz.db')
# 1. `streaming_data` - Contains information about the streaming sessions, including `title_id`, `bandwidth`, `time_interval`, `region`, `resolution`, and `device`.

# 2. `title_data` - Contains details about the titles, including `title_id`, `title_name`, `genre`, and `release_year`.

# 3. `viewership_data` - Contains data about the viewership of titles, including `title_id`, `viewership`, and `time_interval`.

con.sql(
    """
    CREATE TABLE streaming_data (
        title_id VARCHAR,
        bandwidth INT,
        time_interval TIMESTAMP,
        region VARCHAR,
        resolution VARCHAR,
        device VARCHAR
    );
    """
)

con.sql(
    """
    CREATE TABLE title_data (
        title_id VARCHAR,
        title_name VARCHAR,
        genre VARCHAR,
        release_year INT
    );
    """
)

con.sql(
    """
    CREATE TABLE viewership_data (
        title_id VARCHAR,
        viewership INT,
        time_interval TIMESTAMP
    );
    """
)

### Populate tables
con.sql(
    """
    INSERT INTO streaming_data VALUES
    ('tt0001', 100, '2021-01-01 00:00:00', 'US', '720p', 'Mobile'),
    ('tt0002', 200, '2021-01-01 00:00:00', 'US', '1080p', 'Desktop'),
    ('tt0003', 150, '2021-01-01 00:00:00', 'US', '720p', 'Tablet'),
    ('tt0001', 100, '2021-01-01 00:15:00', 'US', '720p', 'Mobile'),
    ('tt0002', 200, '2021-01-01 00:15:00', 'US', '1080p', 'Desktop'),
    ('tt0003', 150, '2021-01-01 00:15:00', 'US', '720p', 'Tablet');
    """
)

con.sql(
    """
    INSERT INTO title_data VALUES
    ('tt0001', 'Title 1', 'Drama', 2020),
    ('tt0002', 'Title 2', 'Comedy', 2019),
    ('tt0003', 'Title 3', 'Action', 2018);
    """
)

con.sql(
    """
    INSERT INTO viewership_data VALUES
    ('tt0001', 1000, '2021-01-01 00:00:00'),
    ('tt0002', 2000, '2021-01-01 00:00:00'),
    ('tt0003', 1500, '2021-01-01 00:00:00'),
    ('tt0001', 1000, '2021-01-01 00:15:00'),
    ('tt0002', 2000, '2021-01-01 00:15:00'),
    ('tt0003', 1500, '2021-01-01 00:15:00');
    """
)