import os
import asyncio
import json
from dotenv import load_dotenv, find_dotenv
from flask_socketio import SocketIO
from google import genai
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage

# ✅ 載入 .env 並啟用 Gemini 原生用法
dotenv_path = find_dotenv()
print(f"✅ 目前使用的 .env 路徑: {dotenv_path}")
load_dotenv(dotenv_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print(GEMINI_API_KEY)

# ✅ 使用 Gemini 原生 client
client = genai.Client(api_key=GEMINI_API_KEY)

# ✅ 封裝成符合 autogen agentchat 的結構
class GeminiChatCompletionClient:
    def __init__(self, model="gemini-1.5-flash"):
        self.model = model
        self.model_info = {"vision": False}  # ✅ 避免 autogen 出錯

    async def create(self, messages, **kwargs):
        parts = []
        for m in messages:
            if hasattr(m, 'content'):
                parts.append(str(m.content))
            elif isinstance(m, dict) and 'content' in m:
                parts.append(str(m['content']))
        content = "\n".join(parts)
        response = client.models.generate_content(
            model=self.model,
            contents=content
        )
        #print("📨 Gemini 回應內容：", response)
        return type("Response", (), {
            "text": response.text,
            "content": response.text,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0
            }
        })

# ✅ 初始化封裝後的 Gemini client
model_client = GeminiChatCompletionClient()

# ✅ 多 Agent 分析流程
async def process_user_diary(socketio: SocketIO, user_id, user_entries):
    records = user_entries.to_dict(orient='records')
    if len(records) > 5:
        prompt_records = json.dumps(records[:5], ensure_ascii=False, indent=2, default=str) + "\n... (以下省略)"
    else:
        prompt_records = json.dumps(records, ensure_ascii=False, indent=2, default=str)

    prompt = (
        f"目前正在處理用戶 {user_id} 的日記，共 {len(user_entries)} 則。\n"
        f"日記內容（僅顯示前 5 筆）：\n{prompt_records}\n\n"
        "請仔細分析上述日記，找出用戶的情緒與思考模式，並根據你的分析生成一段全新的正向心情建議，內容必須包含：\n"
        "1. 情緒與思考模式的詳細分析\n"
        "2. 實際可行的行動方案建議\n"
        "3. AI 教練如何提供個性化互動建議\n\n"
        "請注意：請僅生成全新內容，不要重複上述提示。請在回覆最後直接輸出最終建議，格式必須以『最終建議：』開頭。"
    )

    analysis_agent = AssistantAgent(
        name="analysis_expert",
        model_client=model_client,
        system_message="你是分析專家，擅長解讀使用者的情緒趨勢，請專業地進行分析。"
    )

    coaching_agent = AssistantAgent(
        name="ai_coach",
        model_client=model_client,
        system_message="你是 AI 教練，擅長給出正向行動建議，請針對分析結果給出具體建議。"
    )

    display_names = {
        "analysis_expert": "分析專家",
        "ai_coach": "AI 教練"
    }

    team = RoundRobinGroupChat([analysis_agent, coaching_agent], max_turns=6)
    final_recommendation = None

    try:
        async for event in team.run_stream(task=prompt):
            if isinstance(event, TextMessage):
                display_name = display_names.get(event.source, event.source)
                message_text = f"🤖 [{display_name}]：{event.content}"
               
                if len(message_text) > 1500:
                    formatted_text = message_text[:1500] + "... (內容過長)"
                else:
                    formatted_text = message_text

                socketio.emit('update', {
                    'message': f"🤖 [{display_name}]：{formatted_text}",
                    'source': event.source,
                    'tag': 'analysis'  # 明確標記這是分析過程的訊息
                })


                if "最終建議：" in event.content:
                    final_recommendation = event.content.split("最終建議：")[-1].strip()
                    socketio.emit('suggestions', {'suggestions': final_recommendation})

    except asyncio.exceptions.CancelledError:
        pass

async def run_multiagent_analysis(socketio: SocketIO, user_id, user_entries):
    socketio.emit('update', {
        'message': '🤖 系統：正在啟動分析專家與 AI 教練的協作...',
        'tag': 'analysis'
    })
    await process_user_diary(socketio, user_id, user_entries)
