# Scenario
You work at a company that provides a machine learning model that predicts daily sales for retail stores. The model is trained on historical sales data and is used by store managers to plan inventory and staffing levels. The predictions need to be made every day for the next day's sales. The model takes in historical sales data, weather data, and holiday data as input features.

The current process involves running the model on a local server and generating predictions for each store. The predictions are then sent to the store managers via email. However, as the number of stores and the volume of data increase, the current process is becoming slow and inefficient.

The company wants to move the model to the cloud to improve scalability and performance. The goal is to automate the process of generating predictions for all stores and make the results easily accessible to store managers.

# Requirements
1. The model should be deployed on Google Cloud Platform (GCP).
2. The model should be able to process batch requests for multiple stores at once.
3. The predictions should be stored in a cloud storage bucket and made available to store managers through an API. (Next week we'll make a frontend for the managers)
4. The model should be scheduled to run daily to generate predictions for the next day's sales.

# Assignment: Serving Batch Results of a Model on GCP

## Step-by-Step Instructions
1. Set Up Your GCP Environment

    - Create a new GCP project.
    - Enable the required APIs:
        * Cloud Storage
        * AI Platform
        * Cloud Run
2. Prepare the dataset
    - Ingest the dataset to a Cloud Storage bucket under `gs://<your-bucket-name>/raw_data/`. 
    (`gsutil cp local_dataset.csv gs://<your-bucket-name>/raw_data/`)
    - Hive partition the dataset into batches based on `store_id` and save the partitioned data to `gs://<your-bucket-name>/batch_data/`.

3. Run a ML pipeline
    - Create a Python script that reads the batch data from `gs://<your-bucket-name>/batch_data/`, processes it and writes the results to `gs://<your-bucket-name>/results/`.
    - Create a Docker container that runs the Python script.
    - Deploy the Docker container to Cloud Run.

OR
3. Run Batch Inference on AI Platform

- Use AI Platform Prediction for batch processing:
    * Create a model in AI Platform.
    * Upload your model artifacts (e.g., model.pkl, inference script).
    * Submit a batch prediction job

```bash
gcloud ai-platform jobs submit prediction batch_prediction_job \
    --model=<model-name> \
    --input-paths=gs://<your-bucket-name>/input/ \
    --output-path=gs://<your-bucket-name>/output/ \
    --region=<region>
```

4. Serve batch results to customer via API

```python
from fastapi import FastAPI
from fastapi.responses import FileResponse

app = FastAPI()

@app.get("/results/{store_id}")
def read_item(store_id: int):
    return FileResponse(f"gs://<your-bucket-name>/results/{store_id}.csv")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

5. Schedule Batch Processing of Next Day's Sales at Midnight
    
    - Use Cloud Scheduler to schedule the batch processing job.
    - Set up a cron job to run the job at regular intervals.

```bash
gcloud scheduler jobs create http batch_processing_job \
    --schedule="0 0 * * *" \
    --uri="https://<your-cloud-run-service>/batch_processing" \
    --http-method=GET
```

6. Monitor and Scale

    - Monitor the number of batch jobs failing or succeeding.
    - Use Cloud Monitoring to set up alerts of long-running or failed batch jobs.

```bash
gcloud monitoring alert-policies create \
    --display-name="Batch Job Failure Alert" \
    --condition-display-name="Batch Job Failure" \
    --condition-metric="logging.googleapis.com/user/your-log-metric" \
    --condition-threshold-value="1" \
    --condition-comparison=">"
```

```bash
gcloud monitoring alert-policies create \
    --display-name="Batch Job Duration Alert" \
    --condition-display-name="Batch Job Duration" \
    --condition-metric="logging.googleapis.com/user/your-log-metric" \
    --condition-threshold-value="600" \
    --condition-comparison=">"
```

7. Clean Up
    
    - Delete the GCP project to avoid incurring charges.
    - Delete the Cloud Storage bucket.
    - Delete the AI Platform model.
    - Delete the Cloud Run service.
    - Delete the Docker container image.
    - Delete the Cloud Monitoring alerts.
    - Delete the Cloud Logging logs.
    - Delete the Cloud Billing account.

# Some common issues in real life

* If the model fails to process the next day's sales, what will happen? How can we handle this situation?
 - The model will keep posting yesterday's data without any warning. We can include date in the data to tell the store managers that the data is outdated.
* What would happen if there is a store that has a large number of sales compared to other stores?
    - The model will take longer to process the data for that store. If the data is too large, the model may fail to process the data and then we need to split the data into smaller chunks or allocate more resources for that store.
* How can we ingest new data into the system each day?
    - We can use Cloud Storage to store the new data and then use Cloud Functions to trigger the model to process the new data. Each store has an integration with the Cloud Function to trigger the model.
* How do we find out that the system is not working? What are the key metrics to monitor?
    - We can use Cloud Monitoring to monitor the system. We can monitor the number of failed jobs, the duration of the jobs, and the number of successful jobs. We can also monitor the number of stores that have not been processed.
