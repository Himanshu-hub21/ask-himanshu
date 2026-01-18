import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class LLMHandler:
    """
    Handle LLM calls via Groq (FREE API)
    """
    
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        """
        Initialize Groq client
        
        Available models (FREE):
        - llama-3.1-8b-instant: Fast, good quality
        - llama-3.1-70b-versatile: Better quality, slower
        - mixtral-8x7b-32768: Long context, good reasoning
        """
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model
    
    def generate_response(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """
        Generate response using Groq
        """
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful AI assistant that answers questions "
                            "about Himanshu Ramteke based on provided context. "
                            "Be specific, accurate, and concise. If information is "
                            "not in the context, say so politely."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            response = chat_completion.choices[0].message.content
            return response.strip()
        
        except Exception as e:
            print(f"Error calling Groq API: {e}")
            return "I apologize, but I encountered an error generating a response. Please try again."
