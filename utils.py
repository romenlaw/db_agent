import os
import requests
import re
from bs4 import BeautifulSoup
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# CBA GenAI Studio API configuration
GENAI_API_URL = os.getenv('GENAI_API_URL')
GENAI_API_KEY = os.getenv('GENAI_API_KEY')
CHAT_MODEL='gpt-4o_v2024-05-13_NOFILTER_GaaS'
client=openai.OpenAI(api_key=GENAI_API_KEY, base_url=GENAI_API_URL, timeout=300)

def get_available_models():
    """Get list of available models from the API."""
    headers = {
        'Authorization': f'Bearer {GENAI_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    response = requests.get(
        f'{GENAI_API_URL}/v1/models',
        headers=headers
    )
    
    if response.status_code == 200:
        return response.json()['data']
    else:
        raise Exception(f"Error getting models: {response.text}")
    
def get_available_emb_models():
    """Get list of available embedding models from the API"""
    all_models = get_available_models()
    emb_models = [datum['id'] for datum in all_models if 'emb' in datum['id']]
    return emb_models

def get_basename_without_extension(file_path):
    # Extract the basename (filename with extension)
    base_name = os.path.basename(file_path)
    # Split the base name and the extension
    name, _ = os.path.splitext(base_name)
    return name
