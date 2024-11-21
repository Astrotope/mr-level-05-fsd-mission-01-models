# Deep Learning Models for AI Vehicle Identification/Classification

## Model 01: Azure Custom Vision Endpoint with Azure CV General (compact) [S1]  Model

- Model Description:
  - Azure Custom Vision - General (compact) [S1]
    - Exportable to TF (TensorFlow), TFLite, TFLite-fp16, TFJS (also other formats for iOS and Windows)
    - Deployable in Python, Docker, Javascript, and on Android
- Model Traiing and Validation
  - Dataset Description [GitHub Repo - AI Vehicle Identification Datasets](https://github.com/Astrotope/mr-level-05-fsd-mission-01-datasets)
  - Dataset Archive [Google Drive - ai-vehicle-id-dataset.zip](https://drive.google.com/file/d/1o8ZxFqylNY37aoDljaFLhQDxv_iu9PdI/view?usp=drive_link)
- Model Training/Validation - Google Colab Notebook - Azure Custom Vision SDK/API
  - Training Notebook [Google Colab - Custom Vision Training Notebook](mr_level_05_fsd_mission_01_ai_id_train_cv_model.ipynb)
    - [Launch Notebook in Colab](https://colab.research.google.com/drive/1VAEKGBNkQxk8TRcKKLtTfNUpM8_Jj-Rl?usp=sharing)

## Model 02: Azure ML Endpoint with Google EfficientNet B1 Model (Also testetd MobileNet and ResNet-50. EfficientNet was the best performer.)

- Model Description:
  - Google EfficientNet B1
  - Exportable to TF (PB), TFLite-fp16, TFlite-int8
- Model Traiing and Validation
  - Dataset Description [GitHub Repo - AI Vehicle Identification Datasets](https://github.com/Astrotope/mr-level-05-fsd-mission-01-datasets)
  - Dataset Archive [Google Drive - ai-vehicle-id-dataset.zip](https://drive.google.com/file/d/1o8ZxFqylNY37aoDljaFLhQDxv_iu9PdI/view?usp=drive_link)
- Model Training/Validation - Google Colab Notebook - TensorFlow and Keras
  - Google Colab Notebook.
- Model Deployment
  - Wrapped in an API using FastAPI (Python)
  - Containerized using Docker
  - Pushed to Azure Container Repository
    - Note: Azure ML Endpoint needs Pull Permission from Azure Container Repository. Set this up in Azure Portal - Azure Container Repository
  - Deployed to Provisioned Azure ML Endpoint using Python Deployment Script.
