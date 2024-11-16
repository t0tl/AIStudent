For this week, we will be working with data wrangling. Data wrangling is the process of cleaning, structuring and enriching raw data into a desired format for better decision making in less time. It is a time-consuming process and usually a big chunk of the time in a data science project.

Unless stated otherwise, you can choose to work with one of the following libraries: Pandas, Polars or DuckDB.

# Part 1
## Story:
At Netflix competitor, BingeBlitz, a sudden influx of new subscribers has caused a massive increase in demand, leading to noticeable latency and occasional buffering issues during peak streaming hours. The backend engineer, Alex, is tasked with optimizing the service to handle the load more efficiently. To make informed decisions, Alex seeks insights from you.

## Task 1: Aggregations
Which titles are consuming the most bandwidth during peak hours?
(Objective: Identify high-demand content and evaluate if caching or CDNs can prioritize these titles.)

What are the peak usage times across different regions?
(Objective: Plan for regional load balancing and optimize server resource allocation.)

What are the most common resolutions and devices being used?
(Objective: Determine if adaptive bitrate streaming can be fine-tuned for specific device profiles or resolutions.)

## Task 2: Joins
Are there specific genres or types of content that spike during particular hours or days?
(Objective: Prepare for predictable demand patterns by preloading data or increasing server capacity temporarily.)

## Task 3: Window Functions
Which titles have the fastest growth in viewership per day in the last week?
(Objective: To identify content that is rapidly gaining popularity and allow server capacity to be scaled up.)

```sql
SELECT 
    title_id,
    title_name,
    viewership,
    viewership - LAG(viewership) OVER (PARTITION BY title_id ORDER BY time_interval) AS viewership_growth
FROM viewership_data
WHERE time_interval BETWEEN '2024-11-15' AND '2024-11-22';
```

# Part 2: Analyzing a chronic obstructive pulmonary disease (COPD) dataset

## Scenario:
You work at the World Health Organization and for a future report about COPD, you want to compare the deaths per 100k people in Uganda and the United States during 2019 on a fair basis which takes age distribution into account. The age distribution of each country is important to factor in as we have 

You have a duckdb database `who.db`, which has 3 tables:

1. Population estimates for every age 0-100, from 1950-2021, July 1 for every country in the United Nations.
2. WHO estimates for mortality rate of COPD for 5-year buckets, 0-85+.
3. Standard world population according to: Ahmad OB, Boschi-Pinto C, Lopez AD, Murray CJ, Lozano R, Inoue M (2001). Age standardization of rates: a new WHO standard

## Task 1.
Compare the deaths per 100k people in Uganda and the United States during 2019 using the WHO standard from Lozano R, Inoue M (2001).

## Task 2.
What would be special with Namibia when you wrangle it? (Inspect it's row in a pandas dataframe).