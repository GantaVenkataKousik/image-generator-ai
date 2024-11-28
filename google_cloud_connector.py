import os
from google.cloud import storage, aiplatform

def initialize_google_cloud():
    """
    Initializes Google Cloud services: Storage and AI Platform.
    """
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./service-account-file.json"
    storage_client = storage.Client()
    print("Google Cloud Storage client initialized.")
    
    aiplatform.init(project='your-project-id', location='your-region')
    print("AI Platform client initialized.")

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """
    Uploads a file to a Google Cloud Storage bucket.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name} in bucket {bucket_name}.")

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """
    Downloads a file from a Google Cloud Storage bucket.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)
    print(f"File {source_blob_name} downloaded to {destination_file_name} from bucket {bucket_name}.")

def deploy_model_to_ai_platform(model_name, model_path, serving_container_image_uri):
    """
    Deploys a trained GAN model to Google Cloud AI Platform for online predictions.
    """
    # Upload the model to GCS
    bucket_name = 'your-model-bucket'
    model_blob_name = f"{model_name}/model"
    upload_to_gcs(bucket_name, model_path, model_blob_name)

    # Deploy the model to AI Platform
    endpoint = aiplatform.Endpoint.create(display_name=model_name)
    model = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=f"gs://{bucket_name}/{model_blob_name}",
        serving_container_image_uri=serving_container_image_uri
    )
    model.deploy(endpoint=endpoint, deployed_model_display_name=f"{model_name}-endpoint")
    print(f"Model {model_name} deployed to AI Platform endpoint.")

def generate_images_with_deployed_model(endpoint_name, latent_vector):
    """
    Sends a request to the deployed GAN model on AI Platform for image generation.
    """
    endpoint = aiplatform.Endpoint(endpoint_name=endpoint_name)
    prediction = endpoint.predict(instances=[latent_vector.tolist()])
    print("Prediction from GAN model received.")
    return prediction
initialize_google_cloud()
upload_to_gcs("your-model-bucket", "./trained_model.h5", "gan/trained_model.h5")
download_from_gcs("your-model-bucket", "gan/trained_model.h5", "./downloaded_model.h5")
deploy_model_to_ai_platform(
    model_name="GAN_Model",
    model_path="./trained_model.h5",
    serving_container_image_uri="gcr.io/your-project-id/gan-serving-container"
)
import numpy as np
latent_vector = np.random.normal(0, 1, (1, 100))  # Example latent vector
generated_images = generate_images_with_deployed_model("GAN_Model-endpoint", latent_vector)