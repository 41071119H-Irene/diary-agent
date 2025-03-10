import os
import pandas as pd
import gradio as gr
import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()
conversation_log = []

# 讀取環境變數中的 API Key
gemini_api_key = os.getenv("GEMINI_API_KEY")
model_client = OpenAIChatCompletionClient(model="gemini-2.0-flash", api_key=gemini_api_key)
# 對話終止條件
termination_condition = TextMentionTermination("terminate")
    
# CSV 摘要函式
def summarize_csv(file_path, chunk_size=500):
    summaries = []
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
        summary = f"【區塊 {i+1}】\n前3行：\n{chunk.head(3).to_csv(index=False)}\n總行數：{chunk.shape[0]}"
        summaries.append(summary)
    return "\n".join(summaries)

async def process_file(file_obj, chat_history):
    global conversation_log
    conversation_log.clear()
    if hasattr(file_obj, "name"):
        file_path = file_obj.name
    else:
        chat_history.append({"role": "system", "content": "無法取得檔案路徑"})
        yield chat_history, None
        return
    
    chat_history.append({"role": "system", "content": "📊 開始分析日記內容..."})
    yield chat_history, None

    chunks = list(pd.read_csv(file_path, chunksize=500))
    total_records = sum(chunk.shape[0] for chunk in chunks)
    
    # Debug 訊息
    print("已成功讀取 CSV 檔案，開始處理每個區塊...")

    for idx, chunk in enumerate(chunks):
        chat_history.append({"role": "system", "content": f"📌 正在處理第 {idx+1} 批資料..."})
        yield chat_history, None
        
        prompt = f"第 {idx+1} 批次日記內容:\n{chunk.head(3).to_dict(orient='records')}\n請提供分析與建議。"
        
        agents = [
            AssistantAgent("data_agent", model_client),
            AssistantAgent("analysis_agent", model_client),
            AssistantAgent("coaching_agent", model_client),
            UserProxyAgent("user_proxy")
        ]
        
        team = RoundRobinGroupChat(agents, termination_condition)
        async for event in team.run_stream(task=prompt):
            if isinstance(event, TextMessage):
                chat_history.append({"role": "assistant", "content": event.content})
                conversation_log.append({"source": event.source, "content": event.content})
                
                print(f"💬 Assistant 回應（區塊 {idx+1}）: {event.content}")  # Debug
                yield chat_history, None  # **確保 UI 會即時更新 AI 回應**

    # 分析完成，儲存對話紀錄
    log_file = "positive_thinking_log.csv"
    pd.DataFrame(conversation_log).to_csv(log_file, index=False, encoding="utf-8-sig")
    chat_history.append({"role": "system", "content": "✅ 分析完成！"})
    print("🎯 所有區塊分析完成！")
    yield chat_history, log_file


def send_user_msg(msg, chat_history):
    chat_history.append({"role": "user", "content": msg})
    return chat_history, ""

with gr.Blocks() as demo:
    gr.Markdown("### 📖 正向思考日記分析系統")
    file_input = gr.File(label="上傳 CSV")
    chat_display = gr.Chatbot(label="分析過程", type="messages")
    download_log = gr.File(label="下載分析結果")
    start_btn = gr.Button("開始分析")
    start_btn.click(
        fn=process_file,
        inputs=[file_input, chat_display],
        outputs=[chat_display, download_log]
        )

demo.queue().launch()
