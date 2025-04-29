import asyncio
import json
import pandas as pd
from flask_socketio import SocketIO

from mcp import ModelClient, ProtocolAgent
from materials import 推薦教材根據心情分數, 可選輔導方法

async def process_user_diary(socketio: SocketIO, user_id, user_entries: pd.DataFrame, is_counselor: bool):
    model_client = ModelClient()

    analysis_agent = ProtocolAgent(
        name="analysis_expert",
        role="分析專家",
        model_client=model_client
    )
    coaching_agent = ProtocolAgent(
        name="ai_coach",
        role="AI 教練",
        model_client=model_client
    )

    display_names = {
        "analysis_expert": "分析專家",
        "ai_coach": "AI 教練"
    }

    records = user_entries.to_dict(orient='records')
    if len(records) > 5:
        prompt_records = json.dumps(records[:5], ensure_ascii=False, indent=2, default=str) + "\n... (以下省略)"
    else:
        prompt_records = json.dumps(records, ensure_ascii=False, indent=2, default=str)

    # ✅ 先計算心情指數平均
    try:
        mood_scores = user_entries['心情指數'].astype(float)
        avg_score = mood_scores.mean()
        material_recommendation = 推薦教材根據心情分數(avg_score)
    except Exception as e:
        material_recommendation = "無法計算心情指數，未推薦教材。"

    # ✅ 新 prompt，加上列點式要求
    prompt = (
        f"目前正在處理用戶 {user_id} 的日記，共 {len(user_entries)} 則。\n"
        f"日記內容（僅顯示前 5 筆）：\n{prompt_records}\n\n"
        "請分析用戶的情緒與思考模式，並提出行動建議。\n"
        "要求：\n"
        "- 請用條列式列出分析與建議，每一點簡短清楚（30字內）。\n"
        "- 僅列重點，不要贅述。\n"
        "請以『最終建議：』開頭，後面列出各點。"
    )

    agents = [analysis_agent, coaching_agent]
    all_logs = []
    final_recommendation = None

    try:
        for _ in range(6):
            for agent in agents:
                response = await agent.act(prompt)
                display_name = display_names.get(agent.name, agent.name)

                socketio.emit('update', {
                    'message': f"🤖 [{display_name}]：{response}",
                    'source': agent.name,
                    'tag': 'analysis'
                })
                all_logs.append(response)

                if "最終建議：" in response and final_recommendation is None:
                    final_recommendation = response.split("最終建議：")[-1].strip()

    except asyncio.exceptions.CancelledError:
        pass

    # ✅ 輔導老師要多加選擇最適合的方法（整份日記內容）
    therapy_recommendation = ""
    if is_counselor:
        combined_text = "\n".join([str(r) for r in records])
        therapy_prompt = (
            f"以下是學生的日記內容摘要：\n{combined_text}\n\n"
            "請從以下治療方法中挑選最適合的唯一一種：\n"
            "認知行為治療（CBT）、情緒取向治療（EFT）、解決導向短期治療（SFBT）、正念減壓療法（MBSR）、敘事治療（Narrative Therapy）\n"
            "請回答：建議使用 XX 方法，因為...(50字內)"
        )
        therapy_agent = ProtocolAgent(
            name="therapy_selector",
            role="心理治療建議助手",
            model_client=model_client
        )
        therapy_recommendation = await therapy_agent.act(therapy_prompt)

    # ✅ 組合最終建議
    suggestion_message = ""
    if final_recommendation:
        # 把條列式建議每一點換行
        points = final_recommendation.split('-')
        points = [p.strip() for p in points if p.strip()]
        suggestion_message += "\n".join([f"• {p}" for p in points])

    suggestion_message += f"\n\n📚 {material_recommendation}"
    if therapy_recommendation:
        suggestion_message += f"\n\n🩺 {therapy_recommendation.strip()}"

    socketio.emit('suggestions', {'suggestions': suggestion_message})

async def run_multiagent_analysis(socketio: SocketIO, user_id, user_entries: pd.DataFrame, is_counselor=False):
    socketio.emit('update', {
        'message': '🤖 系統：正在啟動分析專家與 AI 教練的協作...'
    })
    await process_user_diary(socketio, user_id, user_entries, is_counselor)
