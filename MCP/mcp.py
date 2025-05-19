from config import DEFAULT_MODEL, MODEL_PROVIDER, GEMINI_API_KEY, OPENAI_API_KEY, HF_API_KEY
from google import genai
import openai
import requests

# ✅ 多模型支援 Client
class ModelClient:
    default_client = None  # ✅ 可供外部注入的 Gemini Client

    def __init__(self, model=DEFAULT_MODEL, provider=MODEL_PROVIDER):
        self.model = model
        self.provider = provider

        if provider == 'gemini':
            if ModelClient.default_client:
                self.client = ModelClient.default_client
            else:
                self.client = genai.Client(api_key=GEMINI_API_KEY)

        elif provider == 'openai':
            openai.api_key = OPENAI_API_KEY

        elif provider == 'hf':
            self.client = HF_API_KEY  # Hugging Face token

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def generate(self, messages: list):
        content = "\n".join(messages)

        if self.provider == 'gemini':
            response = self.client.models.generate_content(model=self.model, contents=content)
            return response.text.strip()

        elif self.provider == 'openai':
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": content}]
            )
            return response.choices[0].message.content.strip()

        elif self.provider == 'hf':
            url = f"https://api-inference.huggingface.co/models/{self.model}"
            headers = {"Authorization": f"Bearer {self.client}"}
            payload = {"inputs": content}
            r = requests.post(url, headers=headers, json=payload)
            r.raise_for_status()
            return r.json()[0]['generated_text'].strip()

        else:
            raise ValueError("Invalid model provider")

# ✅ 管理上下文訊息
class ContextManager:
    def __init__(self):
        self.history = []

    def add_message(self, role, content):
        self.history.append(f"[{role}] {content}")

    def get_context(self):
        return [msg for msg in self.history]

# ✅ 單一 Agent 封裝
class ProtocolAgent:
    def __init__(self, name, role, model_client: ModelClient):
        self.name = name
        self.role = role
        self.model_client = model_client
        self.context_manager = ContextManager()

    async def act(self, input_text):
        self.context_manager.add_message(self.role, input_text)
        context = self.context_manager.get_context()
        response = await self.model_client.generate(context)
        self.context_manager.add_message(self.name, response)
        return response
