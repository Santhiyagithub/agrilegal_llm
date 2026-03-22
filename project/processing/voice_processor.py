import whisper
import os
import requests
import base64
from dotenv import load_dotenv

load_dotenv()

class VoiceProcessor:
    def __init__(self, model_size="base"):
        print(f"Loading Fallback Whisper model ({model_size})...")
        try:
            self.model = whisper.load_model(model_size)
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            self.model = None

        self.bhashini_api_key = os.getenv("BHASHINI_API_KEY")
        self.bhashini_user_id = os.getenv("BHASHINI_USER_ID")

    def _bhashini_asr(self, file_path: str) -> str:
        if not self.bhashini_api_key:
            raise ValueError("No Bhashini API Key found in .env")
        
        url = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.bhashini_api_key,
            "userID": self.bhashini_user_id or ""
        }
        
        with open(file_path, "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
            
        payload = {
            "pipelineTasks": [{"taskType": "asr", "config": {"language": {"sourceLanguage": "en"}}}], 
            "inputData": {"audio": [{"audioContent": audio_base64}]}
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data["pipelineResponse"][0]["output"][0]["source"]

    def transcribe_audio(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            return "Error: Audio file not found."

        try:
            print(f"Trying Primary: Bhashini ASR for {file_path}...")
            return self._bhashini_asr(file_path)
        except Exception as e:
            print(f"Bhashini ASR failed: {e}. Falling back to Whisper.")
            
            if not self.model:
                return "Error: Bhashini failed and Whisper fallback is not loaded."
                
            try:
                result = self.model.transcribe(file_path)
                transcribed_text = result["text"].strip()
                print(f"Whisper Transcription result: {transcribed_text}")
                return transcribed_text
            except Exception as ex:
                print(f"Error during Whisper fallback: {ex}")
                return "Error: Failed to process audio via primary and fallback engines."
