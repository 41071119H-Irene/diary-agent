import os
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient


load_dotenv()

chat_client = OpenAIChatCompletionClient(
    model="gemini-1.5-flash-latest",  # 或其他你有權限的 Gemini 模型
    api_key=os.getenv("GEMINI_API_KEY")
)

def generate_chat_response(user_message):
    try:
        messages = [
            {"role": "system", "content": "你是一位關心使用者情緒的 AI 教練，請以溫柔、鼓勵的語氣回應。"},
            {"role": "user", "content": user_message}
        ]
        response = chat_client.chat(messages)
        return response["content"].strip()  # ✅ 正確取得 Gemini 回傳的內容
    except Exception as e:
        return f"❌ 發生錯誤：{str(e)}"
