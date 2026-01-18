import re
import os
from dotenv import load_dotenv

load_dotenv()

class PrivacyFilter:
    """
    Detect and filter PII requests/responses
    """
    
    def __init__(self):
        self.owner_email = os.getenv("OWNER_EMAIL", "hsramteke21@gmail.com")
        self.owner_phone = os.getenv("OWNER_PHONE", "REDACTED")
        
        # PII detection patterns
        self.pii_request_patterns = [
            r'\b(phone|mobile|contact|number|call|whatsapp)\b',
            r'\b(address|location|residence|living)\b',
            r'\b(aadhar|pan|passport|ssn)\b',
        ]
        
        self.phone_patterns = [
            r'(\+91|0)?[6-9]\d{9}',
            r'\d{3}[-.]?\d{3}[-.]?\d{4}'
        ]
    
    def is_pii_request(self, query: str) -> bool:
        """
        Check if query asks for PII
        """
        query_lower = query.lower()
        
        for pattern in self.pii_request_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def handle_pii_request(self, query: str) -> str:
        """
        Generate polite response for PII requests
        """
        query_lower = query.lower()
        
        if 'phone' in query_lower or 'mobile' in query_lower or 'contact' in query_lower:
            return (
                f"For any inquiries or to get in touch, please feel free to "
                f"email me at **{self.owner_email}**. I'd be happy to connect!"
            )
        
        elif 'address' in query_lower or 'location' in query_lower:
            return (
                f"I'm currently based in Pune, Maharashtra, India. "
                f"For detailed discussions, please reach out via email: {self.owner_email}"
            )
        
        else:
            return (
                f"I appreciate your interest! For personal information, "
                f"please contact me at {self.owner_email}. Thank you!"
            )
    
    def redact_pii_from_text(self, text: str) -> str:
        """
        Remove phone numbers from text
        """
        for pattern in self.phone_patterns:
            text = re.sub(pattern, '[CONTACT VIA EMAIL]', text)
        
        return text