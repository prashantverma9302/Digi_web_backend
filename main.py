import os
import json
import requests
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS Configuration - Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
WEATHER_API_BASE = 'https://api.weatherapi.com/v1'

# Initialize Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Warning: GEMINI_API_KEY not found in .env")

class ChatRequest(BaseModel):
    prompt: str
    image: Optional[str] = None
    language: str = 'en'

@app.get("/")
def read_root():
    return {"status": "Digital Krishi Backend Running"}

@app.get("/api/weather")
def get_weather(location: str):
    if not WEATHER_API_KEY:
        raise HTTPException(status_code=500, detail="Weather API Key missing on server")
    
    try:
        url = f"{WEATHER_API_BASE}/forecast.json?key={WEATHER_API_KEY}&q={location}&days=3&aqi=no&alerts=no"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Weather API Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch weather")

@app.post("/api/chat")
def chat_with_ai(request: ChatRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key missing on server")

    try:
        # FIX: Use 'gemini-1.5-flash' which is the current stable model name.
        # Avoid 'latest' aliases or experimental versions to prevent 404 errors.
        model = genai.GenerativeModel("models/gemini-2.5-flash-lite")



        
        # Language Customization
        lang_instruction = "Reply in English."
        if request.language == 'hi':
            lang_instruction = "Reply in Hindi using Devanagari script. Use simple farming terminology."
        elif request.language == 'kn':
            lang_instruction = "Reply in Kannada (ಕನ್ನಡ). Use simple farming terminology."
        elif request.language == 'te':
            lang_instruction = "Reply in Telugu (తెలుగు). Use simple farming terminology."

        system_instruction = f"""You are a Digital Krishi Officer (Agricultural Expert). 
        Your goal is to help Indian farmers with:
        1. Crop disease identification.
        2. Treatment recommendations.
        3. Fertilizer advice.
        4. General farming tips.
        
        {lang_instruction}
        
        Format your response nicely with headings and bullet points using Markdown. 
        Keep the tone helpful, encouraging, and easy to understand."""

        # Construct Prompt
        final_prompt = [system_instruction, request.prompt]
        
        # Handle Image
        if request.image:
            try:
                # Basic handling for base64 images
                # Strip header if present (e.g., "data:image/jpeg;base64,")
                image_data = request.image
                if "base64," in image_data:
                    image_data = image_data.split("base64,")[1]
                
                content_parts = [
                    {"mime_type": "image/jpeg", "data": image_data},
                    system_instruction + "\n\n" + request.prompt
                ]
                
                response = model.generate_content(content_parts)
                return {"response": response.text}
                
            except Exception as img_err:
                print(f"Image processing error: {img_err}")
                return {"response": "Error processing image. Please try text only or check the image format."}

        # Text Only
        response = model.generate_content(final_prompt)
        return {"response": response.text}

    except Exception as e:
        print(f"Gemini API Error: {e}")
        # Return a generic error to client but log specific one
        raise HTTPException(status_code=500, detail=f"AI Service Error: {str(e)}")
    
@app.get("/api/models")
def list_models():
    try:
        models = genai.list_models()
        return {"models": [m.name for m in models]}
    except Exception as e:
        return {"error": str(e)}
