import os
from langchain_anthropic import ChatAnthropic

class AnthropicModel:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AnthropicModel, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.api_key = os.getenv('ANTHROPIC_KEY')
        if self.api_key:
            os.environ["ANTHROPIC_API_KEY"] = self.api_key
        else:
            raise ValueError("ANTHROPIC_KEY environment variable is not set.")
        
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20240620",
            temperature=0,
            max_tokens=1024,
            timeout=None,
            max_retries=2,
        )

    def get_llm(self) -> ChatAnthropic:
        return self.llm

if __name__ == "__main__":
    anthropic_model = AnthropicModel()
    llm = anthropic_model.get_llm()
    print(llm.invoke("Hello, how are you?"))

