from retrieval.retriever import LegalRetriever
from inference.llm_engine import LLMEngine
from validation.response_validator import ResponseValidator
from processing.translator import Translator
from processing.voice_processor import VoiceProcessor

class AdvisoryPipeline:
    def __init__(self, llm_model_name="llama3", distance_threshold=0.35):
        self.retriever = LegalRetriever(distance_threshold=distance_threshold)
        self.llm_engine = LLMEngine(model_name=llm_model_name)
        self.validator = ResponseValidator()
        self.translator = Translator()
        self.voice_processor = VoiceProcessor()

    def process_language(self, query: str) -> tuple[str, str]:
        """
        Node 2: Language Processing
        Handle multilingual input via Translator.
        Returns: (normalized_english_query, original_language_code)
        """
        lang_code = self.translator.detect_language(query)
        eng_query = self.translator.translate_to_english(query, lang_code)
        print(f"Language Processing: Original ({lang_code}) -> Normalized ({eng_query})")
        return eng_query, lang_code

    def format_response(self, raw_response: str) -> dict:
        """
        Node 9: Response Formatter
        Parses structured text into dictionary for API JSON output.
        """
        lines = raw_response.split('\n')
        formatted = {
            "provision": "",
            "penalty": "",
            "action": "",
            "source": "",
            "raw": raw_response
        }
        
        current_key = None
        for line in lines:
            line_lower = line.lower()
            if line_lower.startswith("legal provision:"):
                current_key = "provision"
                formatted[current_key] = line.split(":", 1)[1].strip()
            elif line_lower.startswith("penalty:"):
                current_key = "penalty"
                formatted[current_key] = line.split(":", 1)[1].strip()
            elif line_lower.startswith("required action:"):
                current_key = "action"
                formatted[current_key] = line.split(":", 1)[1].strip()
            elif line_lower.startswith("source:"):
                current_key = "source"
                formatted[current_key] = line.split(":", 1)[1].strip()
            elif current_key and line.strip():
                formatted[current_key] += " " + line.strip()
                
        return formatted

    def run(self, input_data: str, is_audio: bool = False):
        """
        Main pipeline entry point.
        If is_audio=True, input_data is a local path to the audio file.
        """
        query = input_data
        if is_audio:
            query = self.voice_processor.transcribe_audio(input_data)
            if query.startswith("Error"):
                return {"status": "error", "message": query}

        # Node 2: Language Processing
        normalized_query, source_lang = self.process_language(query)
        
        # Node 3 & 4 & 5 (Gate 1): Query Embedding + Retrieval + Evidence Confidence
        passed_gate1, context_result = self.retriever.retrieve_and_validate(normalized_query)
        
        if not passed_gate1:
            print(f"[Fallback] Gate 1 Confidence Failed ({context_result}). Falling back to general AI.")
            raw_response = self.llm_engine.generate_general_response(normalized_query)
            final_reply = self.translator.translate_from_english(raw_response, source_lang)
            return {
                "status": "success",
                "final_reply": final_reply,
                "provision": "General Knowledge",
                "penalty": "N/A",
                "action": "N/A",
                "source": "LLM Generative Fallback",
                "raw": raw_response
            }
            
        # Node 6 & 7: Context Builder & LLM Inference
        raw_response = self.llm_engine.generate_response(context_result, normalized_query)
        
        # Node 8 (Gate 2): Structured Output Validation
        passed_gate2, validated_text = self.validator.validate(raw_response)
        
        if not passed_gate2:
            rejection_msg = self.translator.translate_from_english(validated_text, source_lang)
            return {"status": "rejected", "reason": "Gate 2 (Output Validation) Failed", "message": rejection_msg}
            
        # Node 9: Response Formatter
        formatted_response = self.format_response(validated_text)
        formatted_response["status"] = "success"
        
        # Format the final checklist reply
        reply_msg = f"*Legal Provision:* {formatted_response['provision']}\n"
        reply_msg += f"*Penalty:* {formatted_response['penalty']}\n"
        reply_msg += f"*Required Action Checklist:*\n- " + "\n- ".join(formatted_response['action'].split(', ')) + "\n"
        reply_msg += f"\n*Source Reference:* {formatted_response['source']}"
        
        # Translate back if needed
        final_reply = self.translator.translate_from_english(reply_msg, source_lang)
        formatted_response["final_reply"] = final_reply
        formatted_response["transcribed_query"] = query
        
        return formatted_response
