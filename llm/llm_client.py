import os
from typing import Optional, List, Dict
from dotenv import load_dotenv
import yaml

load_dotenv()


class LLMClient:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.provider = self.config['llm']['provider']
        self.model = self.config['llm']['model']
        self.temperature = self.config['llm']['temperature']
        self.max_tokens = self.config['llm']['max_tokens']
        
        if self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        elif self.provider == "openai":
            import openai
            self.client = openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate(self, 
                 prompt: str,
                 system_prompt: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None) -> str:
        """Generate text from prompt"""
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        if self.provider == "anthropic":
            messages = [{"role": "user", "content": prompt}]
            
            kwargs = {
                "model": self.model,
                "max_tokens": max_tok,
                "temperature": temp,
                "messages": messages
            }
            
            if system_prompt:
                kwargs["system"] = system_prompt
            
            response = self.client.messages.create(**kwargs)
            return response.content[0].text
        
        elif self.provider == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok
            )
            return response.choices[0].message.content
    
    def generate_with_context(self,
                            query: str,
                            context: str,
                            system_prompt: Optional[str] = None,
                            temperature: Optional[float] = None) -> str:
        """Generate with query and context"""
        prompt = f"Context:\n{context}\n\nQuery: {query}"
        return self.generate(prompt, system_prompt, temperature)
