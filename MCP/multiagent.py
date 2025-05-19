import asyncio
import json
import re
import pandas as pd
from flask_socketio import SocketIO

from mcp import ModelClient, ProtocolAgent
from materials import 推薦教材根據心情分數, 可選輔導方法

async def process_user_diary(socketio: SocketIO, user_id, user_entries: pd.DataFrame, is_counselor: bool):
    # 🧠 2-1. 初始化模型與代理
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

    # 📊 2-2. 預處理日記資料與推薦教材
    records = user_entries.to_dict(orient='records')
    if len(records) > 5:
        prompt_records = json.dumps(records[:5], ensure_ascii=False, indent=2, default=str) + "\n... (以下省略)"
    else:
        prompt_records = json.dumps(records, ensure_ascii=False, indent=2, default=str)

    # ✅ 計算心情指數平均，推薦教材
    try:
        mood_scores = user_entries['心情指數'].astype(float)
        avg_score = mood_scores.mean()
        material_recommendation = 推薦教材根據心情分數(avg_score)
    except Exception as e:
        material_recommendation = "無法計算心情指數，未推薦教材。"

    # 💬 2-3. 生成分析提示詞（prompt）
    # ✅ 分析提示（列點式）
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

    # 🔄 2-4. 多回合代理互動分析
    try:
        for _ in range(6):
            for agent in agents:
                response = await agent.act(prompt)
                display_name = display_names.get(agent.name, agent.name)

                # 處理回應訊息內容是否需要條列化
                formatted_response = response

                if "最終建議：" in response:
                    # 抽取建議段落以列點
                    rec = response.split("最終建議：")[-1].strip()
                    normalized = re.sub(r'[•*]', '-', rec)
                    points = [p.strip() for p in normalized.split('-') if p.strip()]
                    if points:
                        formatted_response = "最終建議：\n" + "\n".join([f"• {p}" for p in points])

                # ✅ 處理含有 **1. **2. 的項目條列
                elif re.search(r'\*\*\d+\.', response):
                    lines = re.split(r'\*\*(\d+)\.\s*', response)
                    # lines: ['', '1', '項目一', '2', '項目二', ...]
                    grouped = [lines[i+2].strip() for i in range(0, len(lines)-2, 2)]
                    if grouped:
                        formatted_response = "\n".join([f"• {p}" for p in grouped])

                elif any(bullet in response for bullet in ['- ', '* ', '• ']):
                    lines = response.split('\n')
                    points = []
                    for line in lines:
                        m = re.match(r'^[-•*]\s*(.+)', line.strip())
                        if m:
                            points.append(m.group(1).strip())
                    if points:
                        formatted_response = "\n".join([f"• {p}" for p in points])

                socketio.emit('update', {
                    'message': f"🤖 [{display_name}]：{formatted_response}",
                    'source': agent.name,
                    'tag': 'analysis'
                })

                all_logs.append(response)

                if "最終建議：" in response and final_recommendation is None:
                    final_recommendation = response.split("最終建議：")[-1].strip()

    except asyncio.exceptions.CancelledError:
        pass

    # 🧩 2-5. 輔導老師專屬分析（心理治療建議）
    # ✅ 若為輔導老師，給予建議輔導方法
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

    # 🧾 2-6. 整理總建議並回傳前端
    # ✅ 組合建議訊息（避免空列點）
    suggestion_message = ""
    if final_recommendation:
        # 統一條列符號並拆解為清單
        normalized_text = re.sub(r'[•*]', '-', final_recommendation)
        points = [p.strip() for p in normalized_text.split('-') if p.strip()]
        if points:
            suggestion_message += "\n" + "\n".join([f"• {p}" for p in points])

    suggestion_message += f"\n\n📚 {material_recommendation}"
    if therapy_recommendation:
        suggestion_message += f"\n\n🩺 {therapy_recommendation.strip()}"

    socketio.emit('suggestions', {'suggestions': suggestion_message})

# 分析流程啟動
async def run_multiagent_analysis(socketio: SocketIO, user_id, user_entries: pd.DataFrame, is_counselor=False):
    socketio.emit('update', {
        'message': '🤖 系統：正在啟動分析專家與 AI 教練的協作...'
    })
    await process_user_diary(socketio, user_id, user_entries, is_counselor)
