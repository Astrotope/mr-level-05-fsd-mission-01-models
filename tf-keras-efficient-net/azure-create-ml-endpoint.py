# import required libraries
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential

# enter details of your Azure Machine Learning workspace
workspace_name = "mr-level-05-fsd-mission-01-ai-ml"
subscription_id = "27dfa5db-72e7-41cc-8e68-be3fd4e7ad01"
resource_group = "mr-level-05-fsd-mission-01-ai-ml"
location = "australiaeast"

# get a handle to the workspace
ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace_name
)

# Define an endpoint name
endpoint_name = "my-endpoint"

# Example way to define a random name
import datetime

endpoint_name = "endpt-" + datetime.datetime.now().strftime("%m%d%H%M%f")

# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name = endpoint_name, 
    description="this is a sample endpoint",
    auth_mode="key"
)

print(endpoint_name)
print(endpoint)

ml_client.online_endpoints.begin_create_or_update(endpoint, local=True)

ml_client.online_endpoints.get(name=endpoint_name, local=True)




