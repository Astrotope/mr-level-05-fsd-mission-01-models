from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
  Environment,
  ManagedOnlineDeployment,
  ManagedOnlineEndpoint,
  OnlineRequestSettings,
)
from azure.identity import DefaultAzureCredential
from datetime import datetime

# enter details of your Azure Machine Learning workspace
WORKSPACE_NAME = "mr-level-05-fsd-mission-01-ai-ml"
SUBSCRIPTION_ID = "27dfa5db-72e7-41cc-8e68-be3fd4e7ad01"
RESOURCE_GROUP = "mr-level-05-fsd-mission-01-ai-ml"
LOCATION = "australiaeast"

ACR_NAME = "astrotope"

INSTANCE_TYPE="Standard_DS1_v2"
IMAGE_NAME="ai-vehicle-id"
IMAGE_TAG="202411181315"

# The credential is required
credential = DefaultAzureCredential()
print("Credentials retrieved.")

# The MLClient configures Azure ML 
ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)
print("ML Client obtained.")

print("Configure endpoint.")
endpoint = ManagedOnlineEndpoint(
    name="ai-vehicle-id-202411181207",  # Choose your own name; Note that it has to be unique across the Azure location (e.g. westeurope)
    auth_mode="key",  # We use a key for authentication
)

print("Create endpoint.")
# Create the endpoint
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

print("Create environment.")
# Configure a model environment
# This configuration must match with how you set up your API`
environment = Environment(
    name=f"{IMAGE_NAME}-env",
    image=f"{ACR_NAME}.azurecr.io/{IMAGE_NAME}:{IMAGE_TAG}",
    inference_config={
        "scoring_route": {
            "port": 8000,
            "path": "/predict",
        },
        "liveness_route": {
            "port": 8000,
            "path": "/health",
        },
        "readiness_route": {
            "port": 8000,
            "path": "/ready",
        },
    },
)

print("Configure deployment.")
# Configure the deployment
deployment = ManagedOnlineDeployment(
    name=f"dp-{datetime.now():%y%m%d%H%M%S}",  # Add the current time to make it unique
    endpoint_name=endpoint.name,
    model=None,
    environment=environment,
    instance_type=INSTANCE_TYPE,
    instance_count=1,  # we only use 1 instance
    request_settings=OnlineRequestSettings(
        request_timeout_ms=180000,
        #max_concurrent_requests_per_instance=1,
        #max_queue_wait_ms=500,
    ),
)

print("Begin create or update deployment. This could take 8-10 minutes. Be patient.")
# create the online deployment.
# Note that this takes approximately 8 to 10 minutes.
# This is a limitation of Azure. We cannot speed it up.
ml_client.online_deployments.begin_create_or_update(deployment).result()

