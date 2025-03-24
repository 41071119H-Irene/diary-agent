import os
import asyncio
import pandas as pd
from dotenv import load_dotenv
import io

# 根據你的專案結構調整下列 import
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

async def process_user_diary(user_id, user_entries, model_client, termination_condition):
    """
    處理單一用戶的日記內容：
      - 分析用戶的情緒模式
      - 提供個人化的正向思考建議
      - 訓練 AI 教練以進行互動回饋
    """
    prompt = (
        f"目前正在處理用戶 {user_id} 的日記內容，共 {len(user_entries)} 則。\n以下為該用戶的日記內容:\n{user_entries.to_dict(orient='records')}\n\n"
        "請根據以上內容進行分析，並提供該用戶專屬的正向思考建議，包含：\n"
        "  1. 分析該用戶的情緒與思考模式；\n"
        "  2. 提供實際可行的行動方案來改善負面情緒；\n"
        "  3. 訓練 AI 教練，使其能夠與該用戶進行個性化互動。"
    )

    # 為該用戶建立新的 agent 與 team 實例
    analysis_agent = AssistantAgent("analysis_agent", model_client)
    coaching_agent = AssistantAgent("coaching_agent", model_client)
    user_proxy = UserProxyAgent("user_proxy")
    
    team = RoundRobinGroupChat(
        [analysis_agent, coaching_agent, user_proxy],
        termination_condition=termination_condition
    )
    
    messages = []
    async for event in team.run_stream(task=prompt):
        if isinstance(event, TextMessage):
            print(f"[{event.source}] => {event.content}\n")
            messages.append({
                "user_id": user_id,
                "source": event.source,
                "content": event.content,
                "type": event.type,
            })
    return messages

async def main():
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        print("請檢查 .env 檔案中的 GEMINI_API_KEY。")
        return

    model_client = OpenAIChatCompletionClient(
        model="gemini-2.0-flash",
        api_key=gemini_api_key,
    )
    
    termination_condition = TextMentionTermination("terminate")
    
    csv_file_path = "user_diary.csv"
    df = pd.read_csv(csv_file_path)
    
    # 重新命名欄位以匹配程式中的 user_id
    df.rename(columns={"用戶ID": "user_id"}, inplace=True)
    
    if "user_id" not in df.columns:
        print("CSV 檔案缺少 `user_id` 欄位，請確認數據格式。")
        return
    
    user_groups = df.groupby("user_id")
    
    tasks = [
        process_user_diary(user_id, user_entries, model_client, termination_condition)
        for user_id, user_entries in user_groups
    ]
    
    results = await asyncio.gather(*tasks)
    all_messages = [msg for batch in results for msg in batch]
    
    df_log = pd.DataFrame(all_messages)
    output_file = "personalized_positive_thinking_log.csv"
    df_log.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"已將個人化建議輸出為 {output_file}")

if __name__ == '__main__':
    asyncio.run(main())
