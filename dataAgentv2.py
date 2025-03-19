import os
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import io
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer

load_dotenv()

def evaluate_mood(text):
    """
    簡單分析日記內容，根據情緒詞彙判定心情指數 (1-10)。
    """
    positive_words = ["快樂", "開心", "幸福", "興奮", "感恩", "喜悅"]
    negative_words = ["難過", "生氣", "憂鬱", "傷心", "絕望", "沮喪"]
    
    score = 5  # 預設為中性
    for word in positive_words:
        if word in text:
            score += 1
    for word in negative_words:
        if word in text:
            score -= 1
    
    return max(1, min(score, 10))  # 確保數值在 1-10 範圍內

async def process_chunk(chunk, start_idx, total_records, model_client, termination_condition):
    """
    處理單一批次資料，並產生心情指數。
    """
    chunk_data = chunk.to_dict(orient='records')
    
    mood_scores = []
    for entry in chunk_data:
        mood_score = evaluate_mood(entry.get("日記內容", ""))
        mood_scores.append(mood_score)
        entry["心情指數"] = mood_score
    
    prompt = (
        f"目前正在處理第 {start_idx} 至 {start_idx + len(chunk) - 1} 筆日記內容（共 {total_records} 筆）。\n"
        f"以下為該批次日記內容:\n{chunk_data}\n\n"
        "請根據以上內容進行分析，並提供正向思考建議。"
    )
    
    data_agent = AssistantAgent("data_agent", model_client)
    analysis_agent = AssistantAgent("analysis_agent", model_client)
    coaching_agent = AssistantAgent("coaching_agent", model_client)
    user_proxy = UserProxyAgent("user_proxy")
    
    team = RoundRobinGroupChat(
        [data_agent, analysis_agent, coaching_agent, user_proxy],
        termination_condition=termination_condition
    )
    
    messages = []
    async for event in team.run_stream(task=prompt):
        if isinstance(event, TextMessage):
            print(f"[{event.source}] => {event.content}\n")
            messages.append({
                "batch_start": start_idx,
                "batch_end": start_idx + len(chunk) - 1,
                "source": event.source,
                "content": event.content,
                "type": event.type,
            })
    
    return messages, mood_scores

def plot_mood_trend(mood_data, output_path="mood_trend.png"):
    """
    繪製心情走向趨勢圖。
    """
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(mood_data) + 1), mood_data, marker='o', linestyle='-')
    plt.xlabel("日記編號")
    plt.ylabel("心情指數 (1-10)")
    plt.title("心情走向趨勢圖")
    plt.grid()
    plt.savefig(output_path)
    plt.show()

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
    chunk_size = 500
    chunks = list(pd.read_csv(csv_file_path, chunksize=chunk_size))
    total_records = sum(chunk.shape[0] for chunk in chunks)
    
    tasks = [
        process_chunk(chunk, idx * chunk_size, total_records, model_client, termination_condition)
        for idx, chunk in enumerate(chunks)
    ]
    
    results = await asyncio.gather(*tasks)
    all_messages = [msg for batch in results for msg in batch[0]]
    all_mood_scores = [score for batch in results for score in batch[1]]
    
    df_log = pd.DataFrame(all_messages)
    df_mood = pd.DataFrame({"日記編號": range(1, len(all_mood_scores) + 1), "心情指數": all_mood_scores})
    
    output_file = "positive_thinking_log.csv"
    mood_file = "mood_analysis.csv"
    df_log.to_csv(output_file, index=False, encoding="utf-8-sig")
    df_mood.to_csv(mood_file, index=False, encoding="utf-8-sig")
    
    print(f"已將所有對話紀錄輸出為 {output_file}")
    print(f"已將心情指數輸出為 {mood_file}")
    
    plot_mood_trend(all_mood_scores)

if __name__ == '__main__':
    asyncio.run(main())
