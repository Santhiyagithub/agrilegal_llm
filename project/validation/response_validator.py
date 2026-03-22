import re

class ResponseValidator:
    def __init__(self):
        # We expect exact headers from the LLM prompt instructions
        self.expected_fields = [
            r"Legal Provision:",
            r"Penalty:",
            r"Required Action:",
            r"Source:"
        ]

    def validate(self, response_text: str):
        """
        Gate 2: Structured Output Validation
        Ensures generated responses follow structured format and contain legal references.
        """
        clean_text = response_text.strip()

        # If LLM correctly identified insufficient context
        if "Insufficient authoritative information available." in clean_text:
            return False, "Insufficient authoritative information available."
            
        # Check if all required fields are present
        for field in self.expected_fields:
            if not re.search(field, clean_text, re.IGNORECASE):
                print(f"[Gate 2 Rejection] Missing structural field: '{field}'")
                return False, "Response rejected due to insufficient validation."

        # Make sure the source has a citation content
        # For simplicity, we split line by line and look at content after the headers
        lines = clean_text.split('\n')
        source_found = False
        
        for line in lines:
            if re.search(r"Source:", line, re.IGNORECASE):
                # Ensure something comes after "Source:"
                parts = line.split(":", 1)
                if len(parts) > 1 and len(parts[1].strip()) > 3: # at least 3 char citation
                    source_found = True
                    break
            
        if not source_found:
            print("[Gate 2 Rejection] Source/Citation missing or too brief.")
            return False, "Response rejected due to insufficient validation."

        print("[Gate 2 Passed] Output format validated.")
        return True, clean_text
