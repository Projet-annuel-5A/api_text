# Text Emotions Analysis module

## Overview
The Text module of the **Interviewz** application processes textual data to analyze emotional content using machine learning models. It leverages state-of-the-art NLP models for sequence classification to predict emotional responses based on the text from interviews.

## Directory Structure
The module consists of several Python files organized as follows:
```plaintext
text/
├── app.py
├── textEmotions.py
├── utils/
│   ├── models.py
│   ├── utils.py
```

## Components

### FastAPI Application (app.py)
Initializes a FastAPI application.

#### API Endpoints

```fastAPI
@app.get("/health")
"""
Returns the health status of the API. 
Description: Endpoint for checking the health status of the application.
Response: Returns a JSON object with the status "ok".
"""
```
```fastAPI
@app.post("/analyse_text")
"""
Endpoint to process text data for emotional analysis from stored interview data.
Parameters:
    session_id (int): The unique identifier for the session.
    interview_id (int): The unique identifier for the interview.
Functionality:
    Retrieves texts from a database.
    Processes each text to determine emotion using a deep learning model.
    Updates the database with the analyzed results.
Response:
    JSON object with the status "ok" upon successful processing.
Raises:
    HTTPException: Exception with a status code 500 indicating a server error if the process fails.
"""
```
```fastAPI
@app.post("/testConfig")
"""
Endpoint for testing the device where the models where loaded.
Response: JSON object showing the model ID and the device (CPU or GPU) it is loaded on.
"""
```

### TextEmotions (textEmotions.py):
Manages the text analysis process by fetching text data, applying emotional analysis, and updating results.
Utilizes pre-trained NLP models to classify text into emotional categories.

### Utilities (utils/utils.py): 
Includes logging setup, configuration management, and methods for file operations on S3 storage.
Implements methods for updating database records and managing connections to Supabase for data storage.

### Models (utils/models.py):

Responsible for loading and managing NLP models and tokenizers for text emotion classification.
Implements singleton pattern to ensure models are loaded once per application lifecycle.