# azure-ml-deploy instructions

### Setup Python virtual environment

```bash
mkdir deploy-azure-ml-keras && cd deploy-azure-ml-keras
mkdir -p venv
python3.10 -m venv venv
source venv/bin/activate
which python3
```

### Create requirements.txt and install requirements
```bash
nano requirements.txt
add...
pip==24.3.1
azure-ai-ml==1.22.1
azure-core==1.32.0
azure-identity==1.19.0
azureml.mlflow==1.58.0.post3
azure-mgmt-containerregistry==10.3.0
azure-mgmt-authorization==4.0.0
azure-mgmt-resource==23.2.0
azure-mgmt-compute==33.0.0
azure-mgmt-core==1.5.0
azure-mgmt-resource==23.2.0
python-dotenv==1.0.1

pip3 install -r requirements.txt
```

### Create Dockerfile FROM python:3.10-slim

```docker
WORKDIR /app

ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ADD api.py .
COPY ai_id_vehicle.keras .

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Create API file api.py

```python
from fastapi import FastAPI
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict

import json
import os
from io import BytesIO

import numpy as np
from PIL import Image

from keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_NAME = 'ai_id_vehicle.keras'
MODEL_VERSION = '1'
MODEL_DIR = os.path.splitext(MODEL_NAME)[0]

def init():

  global model
  global MODEL_NAME
  global MODEL_VERSION
  global MODEL_DIR
  
  # Get the current working directory
  current_directory = os.getcwd()
  print("Current Directory:", current_directory)

  # List the contents of the current directory
  contents = os.listdir(current_directory)
  print("Contents of the Directory:")
  for item in contents:
    print(item)
    
  model_name = MODEL_NAME # os.getenv('MODEL_NAME')
  model_version = MODEL_VERSION # os.getenv('MODEL_VERSION')
  model_label = os.path.splitext(model_name)[0]
  model_path = os.path.join(MODEL_DIR, model_version, model_name)

  # Check if the file exists
  if os.path.exists(model_path):
    print(f"File exists {model_path}.")
  else:
    print(f"File does not exist {model_path}.")

  print(f"model_path: {model_path}")
  
  model = load_model('ai_id_vehicle.keras') #(model_path)
  if model:
    print('Model {model_label} at path {model_path} loaded')
    return True
  return False

def prepare_image(image):

    dimension = 224
    shape = (dimension, dimension)
    # Resize the image, convert to RGB
    image = image.resize(shape, resample=Image.BILINEAR)
    # Convert to numpy array and normalize
    image = np.array(image) / 255.0
    # Reshape to add batch dimension: (1, 224, 224, 3)
    image = image.reshape(1, dimension, dimension, 3)
    return image

app = FastAPI(title="AI Vehicle ID")
model = None
init_success = init()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    global init_success
    if init_success:
        return {"status": "ok"}
    else:
        return {"status": "not ready"}

@app.post("/predict", response_model=None)
async def predict(image: UploadFile):
    try:
        # Read the image bytes from the uploaded file
        image_bytes = await image.read()
        # Open the image using PIL
        pil_image = Image.open(BytesIO(image_bytes))

        # Preprocess the image for the model
        prepared_image = prepare_image(pil_image)

        # Make a prediction using the pre-trained model
        prediction = model.predict(prepared_image)

        # Assuming you have a list `category_tags` to map indices to categories
        category_tags = ["Negative","cab", "convertible", "coupe", "hatchback", "minivan", "sedan", "suv", "truck", "van", "wagon"]

        # Extract the prediction results
        index_max_prob = np.argmax(prediction[0])
        predicted_category = category_tags[index_max_prob]
        prediction_probability = float(prediction[0][index_max_prob])
        predictions_dict = {tag: float(prob) for tag, prob in zip(category_tags, prediction[0])}

        # Create a result dictionary and include the form data
        result_dict = {
            "message": "Image and form data received and processed successfully!",
            "prediction": {
                "category": predicted_category,
                "probability": prediction_probability
            },
            "predictions": predictions_dict
        }

        return JSONResponse(content=result_dict)

    except Exception as e:

        return JSONResponse(content={"error": str(e), "status": "500"}) 
```


### Create new requirements.txt


```bash
mv requirements.txt requirements.txt.venv
cp requirements.txt.venv requirements.txt
nano requirements.txt
```

```python
pip==24.3.1
numpy==1.26.4
fastapi==0.109.2
pydantic==2.9.2
tensorflow==2.17.1
keras==3.5.0
uvicorn
pillow
python-multipart
```

### Copy-in keras model

```bash
cp /Users/cmcewing/Documents/mission_ready/level-05/mission-01/models
/tf-keras-efficient-net/models/vehicle_classification_model_v005-
efficient-net-adam-0.01-30-epochs-transfer-20-epochs-deep-
negative-included.keras ai_id_vehicle.keras
```

### Login to Azure CLI [This takes a moment]
```bash
az login
select ms account in browser window that opens
A web browser has been opened at https://login.microsoftonline.com/organizations/oauth2/v2.0/authorize. Please continue the login in the web browser. If no web browser is available or if the web browser fails to open, use device code flow with `az login --use-device-code`.

Retrieving tenants and subscriptions for the selection...

[Tenant and subscription selection]

No     Subscription name     Subscription ID                       Tenant
-----  --------------------  ------------------------------------  -------------
[1] *  Azure subscription 1  [Your Subscription ID ]  Mission Ready

The default is marked with an *; the default tenant is 'Mission Ready' and subscription is 'Azure subscription 1' ([Your Subscription ID ]).

Select a subscription and tenant (Type a number or Enter for no changes): 1

Tenant: Mission Ready
Subscription: Azure subscription 1 ([Your Subscription ID ])

[Announcements]
With the new Azure CLI login experience, you can select the subscription you want to use more easily. Learn more about it and its configuration at https://go.microsoft.com/fwlink/?linkid=2271236

If you encounter any problem, please open an issue at https://aka.ms/azclibug

[Warning] The login output has been updated. Please be aware that it no longer displays the full list of available subscriptions by default.
```

### Check docker & docker desktop are configured and running

```bash
which docker
/usr/local/bin/docker
docker --version 
Docker version 27.3.1, build ce12230
```

### Login to Azure Container Registry [this takes a moment]

```bash
az acr login --name astrotope
Login Succeeded
```

### Build container [ if date is 2024-11-25 at 08:09am, then image tag is 202411250809]

```bash
docker build -t astrotope.azurecr.io/ai-vehicle-id:202411260703 . --no-cache
```

### Test container locally

##### Run container

```bash
docker run -p 8000:8000 astrotope.azurecr.io/ai-vehicle-id:202411260703
```

```bash
2024-11-25 18:07:20.495400: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2024-11-25 18:07:20.607187: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2024-11-25 18:07:20.775623: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-11-25 18:07:20.898843: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-11-25 18:07:20.919154: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2024-11-25 18:07:21.113514: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-11-25 18:07:23.136127: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1732558088.924619      53 service.cc:146] XLA service 0x7f7494013230 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
I0000 00:00:1732558088.926020      53 service.cc:154]   StreamExecutor device (0): Host, Default Version
2024-11-25 18:08:09.293930: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:268] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
I0000 00:00:1732558094.455543      53 device_compiler.h:188] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
Current Directory: /app
Contents of the Directory:
__pycache__
ai_id_vehicle
ai_id_vehicle.keras
api.py
requirements.txt
File exists ai_id_vehicle/1/ai_id_vehicle.keras.
model_path: ai_id_vehicle/1/ai_id_vehicle.keras
Model {model_label} at path {model_path} loaded
1/1 ━━━━━━━━━━━━━━━━━━━━ 9s 9s/step
INFO:     172.17.0.1:61522 - "POST /predict HTTP/1.1" 200 OK
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 644ms/step
INFO:     172.17.0.1:55504 - "POST /predict HTTP/1.1" 200 OK
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 606ms/step
INFO:     172.17.0.1:58940 - "POST /predict HTTP/1.1" 200 OK
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 642ms/step
INFO:     172.17.0.1:64306 - "POST /predict HTTP/1.1" 200 OK
^CINFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [1]
```

##### Test with postman

Request:

- url: 

  - http://0.0.0.0:8000/predict

- headers: 

  - [none]

- body:

  - form-data: 'image': [image-file-of-negative-background]


Response:

- headers: 

  - 'date': Mon, 25 Nov 2024 18:12:23 GMT
  - 'server': uvicorn
  - 'content-length': 467
  - 'content-type': application/json

- body:

	```json
	{
	    "message": "Image and form data received and processed successfully!",
	    "prediction": {
	        "category": "Negative",
	        "probability": 1.0
	    },
	    "predictions": {
	        "Negative": 1.0,
	        "cab": 1.5778397685628498e-11,
	        "convertible": 2.7224864029840035e-10,
	        "coupe": 4.363242059324257e-11,
	        "hatchback": 1.4754616799173004e-12,
	        "minivan": 1.7155242036778675e-15,
	        "sedan": 2.374586801062034e-13,
	        "suv": 1.322380427382086e-09,
	        "truck": 3.3150197864539876e-11,
	        "van": 3.106431084587413e-11,
	        "wagon": 9.859618546950721e-13
	    }
	}
	```

### Deploy container to Azure Container Registry

```bash
az acr login --name astrotope [just in case login expired]                                                                                                                                                                        
Login Succeeded

docker push astrotope.azurecr.io/ai-vehicle-id:202411260703                                                                                                                                           ─╯
The push refers to repository [astrotope.azurecr.io/ai-vehicle-id]
ebd33dbf84e9: Pushed
384065d45657: Pushed
7721df527a0f: Pushed
13529d9a2838: Pushed
ffb62924d2a1: Pushing [===========================>                       ]  1.211GB/2.229GB
3e1bfd4f2a08: Pushed
f3e22e4563d0: Pushed
1062cd4d071e: Layer already exists
e5d8b619f2ce: Layer already exists
6a4c9147f069: Layer already exists
c3548211b826: Layer already exists
...
202411260703: digest: sha256:cfb59e4c65fa6a60da581f6b03de57d37352996e4e658a5bc31ca2a93a2c6dbe size: 2624
```

### Create Azure ML Endpoint Deployment Script

```bash
nano .env
```

```bash
WORKSPACE_NAME="mr-level-05-fsd-mission-01-ai-ml"
SUBSCRIPTION_ID="27dfa5db-72e7-41cc-8e68-be3fd4e7ad01"
RESOURCE_GROUP="mr-level-05-fsd-mission-01-ai-ml"
LOCATION="australiaeast"

ACR_NAME="astrotope"
ACR_RESOURCE_GROUP="mr-level-05-fsd-cr-group"

INSTANCE_TYPE="Standard_DS1_v2"
IMAGE_NAME="ai-vehicle-id"
IMAGE_TAG="202411260703"
```

```bash
nano deploy-endpoint-ai-vehicle-id.py
```
with contents...

```python
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
```
### Deploy ML Endpoint using deploy-endpoint-ai-vehicle-id.py

```bash
python3 deploy-endpoint-ai-vehicle-id.py
```

```bash
Credentials retrieved.
ML Client obtained.
Configure endpoint: ai-vehicle-id-202411260703
Endpoint configured: ai-vehicle-id-202411260703
Create endpoint.
Readonly attribute principal_id will be ignored in class <class 'azure.ai.ml._restclient.v2022_05_01.models._models_py3.ManagedServiceIdentity'>
Readonly attribute tenant_id will be ignored in class <class 'azure.ai.ml._restclient.v2022_05_01.models._models_py3.ManagedServiceIdentity'>
Managed Identity Client ID: 9d45c3bb-bb61-4063-bd7c-73c32aec1628
Role assignment created successfully: /subscriptions/27dfa5db-72e7-41cc-8e68-be3fd4e7ad01/resourceGroups/mr-level-05-fsd-cr-group/providers/Microsoft.ContainerRegistry/registries/astrotope/providers/Microsoft.Authorization/roleAssignments/fd378b41-eca9-4c04-bbe2-0ab109acaa85
Create environment.
Configure deployment.
Begin create or update deployment. This could take 8-10 minutes. Be patient.
Instance type Standard_DS1_v2 may be too small for compute resources. Minimum recommended compute SKU is Standard_DS3_v2 for general purpose endpoints. Learn more about SKUs here: https://learn.microsoft.com/en-us/azure/machine-learning/referencemanaged-online-endpoints-vm-sku-list
Check: endpoint ai-vehicle-id-202411260703 exists
.........................................................................
```


