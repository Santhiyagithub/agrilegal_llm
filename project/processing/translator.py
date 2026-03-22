import os
import requests
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

load_dotenv()

class Translator:
    def __init__(self):
        print("Initializing Prime Bhashini Translate + DeepTranslator fallback...")
        self.bhashini_user_id = os.getenv("BHASHINI_USER_ID")
        self.bhashini_api_key = os.getenv("BHASHINI_API_KEY")

    def _call_bhashini_api(self, text: str, source: str, target: str, task: str = "translation") -> str:
        if not self.bhashini_api_key:
            raise ValueError("No Bhashini API Key found in .env")
            
        url = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.bhashini_api_key,
            "userID": self.bhashini_user_id or ""
        }
        payload = {
            "pipelineTasks": [{"taskType": task, "config": {"language": {"sourceLanguage": source, "targetLanguage": target}}}],
            "inputData": {"input": [{"source": text}]}
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data["pipelineResponse"][0]["output"][0]["target"]

    def detect_language(self, text: str) -> str:
        # Basic heuristic fallback for detection
        if any(ord(c) > 127 for c in text):
            return "hi"
        return "en"

    def translate_to_english(self, text: str, source_lang: str) -> str:
        if source_lang == "en": return text
        
        try:
            print(f"Trying Primary: Bhashini Translate ({source_lang} -> en)")
            return self._call_bhashini_api(text, source_lang, "en")
        except Exception as e:
            print(f"Bhashini translation failed: {e}. Falling back to Google Translate.")
            return GoogleTranslator(source=source_lang, target='en').translate(text)

    def translate_from_english(self, text: str, target_lang: str) -> str:
        if target_lang == "en": return text
        
        try:
            print(f"Trying Primary: Bhashini Translate (en -> {target_lang})")
            return self._call_bhashini_api(text, "en", target_lang)
        except Exception as e:
            print(f"Bhashini translation failed: {e}. Falling back to Google Translate.")
            return GoogleTranslator(source='en', target=target_lang).translate(text)
