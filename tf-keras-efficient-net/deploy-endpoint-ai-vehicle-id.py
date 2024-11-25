from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
  Environment,
  ManagedOnlineDeployment,
  ManagedOnlineEndpoint,
  ManagedIdentityConfiguration,
  IdentityConfiguration,
  OnlineRequestSettings,
)
from azure.core.exceptions import ResourceExistsError  # Import the exception explicitly
from azure.identity import DefaultAzureCredential
from azure.mgmt.authorization import AuthorizationManagementClient
from azure.mgmt.containerregistry import ContainerRegistryManagementClient
from datetime import datetime
import uuid
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the environment variables
WORKSPACE_NAME = os.getenv("WORKSPACE_NAME")
SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")
RESOURCE_GROUP = os.getenv("RESOURCE_GROUP")
LOCATION = os.getenv("LOCATION")

ACR_NAME = os.getenv("ACR_NAME")
ACR_RESOURCE_GROUP = os.getenv("ACR_RESOURCE_GROUP")

INSTANCE_TYPE = os.getenv("INSTANCE_TYPE")
IMAGE_NAME = os.getenv("IMAGE_NAME")
IMAGE_TAG = os.getenv("IMAGE_TAG")

print(f"Workspace: {WORKSPACE_NAME}, Location: {LOCATION}")

# The credential is required
credential = DefaultAzureCredential()
auth_client = AuthorizationManagementClient(credential, SUBSCRIPTION_ID)
acr_client = ContainerRegistryManagementClient(credential, SUBSCRIPTION_ID)
print("Credentials retrieved.")

# Get the ACR resource ID
acr = acr_client.registries.get(resource_group_name=ACR_RESOURCE_GROUP, registry_name=ACR_NAME)
acr_resource_id = acr.id

# Define the role and scope
role_definition_id = f"/subscriptions/{SUBSCRIPTION_ID}/providers/Microsoft.Authorization/roleDefinitions/7f951dda-4ed3-4680-a7ca-43fe172d538d"  # AcrPull role
scope = acr_resource_id

# The MLClient configures Azure ML 
ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)
print("ML Client obtained.")

ML_ENDPOINT_NAME = f"ai-vehicle-id-{IMAGE_TAG}"
print(f"Configure endpoint: {ML_ENDPOINT_NAME}")
endpoint = ManagedOnlineEndpoint(
    name=f"{ML_ENDPOINT_NAME}",  # Choose your own name; Note that it has to be unique across the Azure location (e.g. westeurope)
    auth_mode="key",  # We use a key for authentication
    identity=IdentityConfiguration(type="SystemAssigned"),  # Enable system-assigned managed identity
    #identity={"type": "SystemAssigned"},  # Enable system-assigned managed identity
    #identity=ManagedIdentityConfiguration(type="SystemAssigned"),  # Enable system-assigned managed identity
)
print(f"Endpoint configured: {ML_ENDPOINT_NAME}")

print("Create endpoint.")
# Create the endpoint
endpoint_result = ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Fetch the created endpoint's identity
endpoint_details = ml_client.online_endpoints.get(name=ML_ENDPOINT_NAME)
ml_managed_identity_client_id = endpoint_details.identity.principal_id  # This is the client ID of the managed identity

print(f"Managed Identity Client ID: {ml_managed_identity_client_id}")

# Assign the role to the ML managed identity
try:
    role_assignment_name = uuid.uuid4()
    role_assignment = auth_client.role_assignments.create(
        scope=scope,
        role_assignment_name=str(role_assignment_name),
        parameters={
            "properties": {
                "roleDefinitionId": role_definition_id,
                "principalId": ml_managed_identity_client_id,  # Identity's client ID
            }
        },
    )
    print(f"Role assignment created successfully: {role_assignment.id}")
except ResourceExistsError:
    print("Role assignment already exists, skipping creation.")
    role_assignment = None  # Explicitly set it as None since it won't exist
except Exception as e:
    print(f"An error occurred: {e}")
    raise

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
