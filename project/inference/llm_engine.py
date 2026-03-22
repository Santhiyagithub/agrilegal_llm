from langchain_core.prompts import PromptTemplate
import ollama

class LLMEngine:
    def __init__(self, model_name="llama3"):
        self.model_name = model_name
        # Context Builder configuration
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Answer ONLY using the provided legal context.
If the information is insufficient to answer the query, say EXACTLY: "Insufficient authoritative information available."

Provide the answer in the following structured format exactly as shown below:

Legal Provision: [Provision or section name here]
Penalty: [Any penalties described here, or "None mentioned"]
Required Action: [Required actions described here, or "None mentioned"]
Source: [Document source here]

Context:
{context}

Question:
{question}"""
        )

    def generate_response(self, context_docs, question: str):
        # Context Builder: merge text from context_docs
        # context_docs is a list of (Document, score) tuples
        context_parts = []
        for index, (doc, score) in enumerate(context_docs):
            source = doc.metadata.get('source', 'Unknown Document')
            context_parts.append(f"--- Document {index+1} (Source: {source}) ---\n{doc.page_content}")
            
        context_text = "\n\n".join(context_parts)
        
        prompt = self.prompt_template.format(context=context_text, question=question)
        
        print(f"Querying local model ({self.model_name}) via Ollama...")
        try:
            response = ollama.chat(model=self.model_name, messages=[
                {"role": "user", "content": prompt}
            ])
            return response['message']['content'].strip()
        except Exception as e:
            print(f"Error querying Ollama: {e}")
            return "Error: Could not communicate with local model."

    def generate_general_response(self, question: str):
        prompt = f"Answer the following question conversationally and helpfully as an AI assistant. You do not need to follow any strict legal formatting.\n\nQuestion: {question}"
        
        print(f"Querying local model ({self.model_name}) for general answer...")
        try:
            response = ollama.chat(model=self.model_name, messages=[
                {"role": "user", "content": prompt}
            ])
            return response['message']['content'].strip()
        except Exception as e:
            print(f"Error querying Ollama for general response: {e}")
            return "Error: Could not communicate with local model."
