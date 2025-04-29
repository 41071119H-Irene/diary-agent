import os 
import asyncio
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.agents.web_surfer import MultimodalWebSurfer

# 載入環境變數
load_dotenv()

async def main():
    # 從 .env 讀取 Gemini API 金鑰
    gemini_api_key = os.environ.get("GEMINI_API_KEY")

    # 初始化 LLM client
    model_client = OpenAIChatCompletionClient(
        model="gemini-2.0-flash",
        api_key=gemini_api_key,
    )

    start_page = "https://www.google.com/"

    # 建立 AI 助理 Agent
    emotion_analyst = AssistantAgent("assistant", model_client)
    web_surfer = MultimodalWebSurfer("web_surfer", model_client, start_page)
    user_proxy = UserProxyAgent("user_proxy")


    # 當對話中出現 "exit" 時即終止對話
    termination_condition = TextMentionTermination("exit")

    # 建立群組對話（只有使用者與助理）
    team = RoundRobinGroupChat(
        [web_surfer, emotion_analyst, user_proxy],
        termination_condition=termination_condition
    )

# 啟動團隊對話，任務是「搜尋 Gemini 的相關資訊，並撰寫一份簡短摘要」
    await Console(team.run_stream(task="請搜尋今日台北天氣並推測對於心情的影響，撰寫一份簡短摘要✨"))

if __name__ == "__main__":
    asyncio.run(main())