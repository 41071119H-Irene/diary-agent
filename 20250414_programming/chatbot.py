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

    # 建立 AI 助理 Agent
    emotion_analyst = AssistantAgent("assistant", model_client)
    web_surfer = MultimodalWebSurfer("web_surfer", model_client)
    user_proxy = UserProxyAgent("user_proxy")


    # 當對話中出現 "exit" 時即終止對話
    termination_condition = TextMentionTermination("exit")

    # 🟡 第一次對話：包含 WebSurfer 做一次搜尋任務
    team_with_search = RoundRobinGroupChat(
        [web_surfer, emotion_analyst, user_proxy],
        termination_condition=termination_condition
    )
    await Console(team_with_search.run_stream(task="請搜尋今日天氣，並用簡短摘要回報"))

    # 🟢 後續對話：只有 Assistant 跟使用者對話
    team_conversation_only = RoundRobinGroupChat(
        [user_proxy, emotion_analyst],
        termination_condition=termination_condition
    )
    await Console(team_conversation_only.run_stream(task="✨ 搜尋完成，請與情緒探索小幫手自由對話吧（輸入 'exit' 離開） ✨"))


if __name__ == "__main__":
    asyncio.run(main())
