import os
import asyncio
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import pandas as pd
from AgentEMOv3 import process_user_diary
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import TextMentionTermination

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("static/moodtrend", exist_ok=True)

load_dotenv()

# 初始化 OpenAI 客戶端
gemini_api_key = os.environ.get("GEMINI_API_KEY")
model_client = OpenAIChatCompletionClient(
    model="gemini-2.0-flash",
    api_key=gemini_api_key
)
termination_condition = TextMentionTermination("terminate")


# **首頁**
@app.route("/")
def index():
    return render_template("index.html", big_3_goals=["待更新"] * 3, recommendations=["待更新"] * 3)


# **上傳 CSV 處理**
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("index"))

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # 分析 CSV 檔案
    df = pd.read_csv(file_path)
    df.rename(columns={"用戶ID": "user_id"}, inplace=True)

    user_groups = df.groupby("user_id")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def process_all():
        tasks = [
            process_user_diary(user_id, user_entries, model_client, termination_condition)
            for user_id, user_entries in user_groups
        ]
        results = await asyncio.gather(*tasks)
        return results

    # results = loop.run_until_complete(process_all())
    results = asyncio.run(process_all())


    # 儲存結果
    all_messages = [msg for batch in results if isinstance(batch, list) for msg in batch]
    df_log = pd.DataFrame(all_messages)
    df_log.to_csv("personalized_positive_thinking_log.csv", index=False, encoding="utf-8-sig")

    return redirect(url_for("index"))


# **取得 AI 生成的建議**
@app.route("/get_recommendations", methods=["GET"])
def get_recommendations():
    try:
        df = pd.read_csv("personalized_positive_thinking_log.csv")
        df = df.dropna()
        
        if df.empty:
            return jsonify({"big_3_goals": ["無數據"] * 3, "recommendations": ["無數據"] * 3})
        
        big_3 = df[df["source"] == "analysis_agent"]["content"].tolist()[:3]
        recommendations = df[df["source"] == "coaching_agent"]["content"].tolist()[:3]

        return jsonify({
            "big_3_goals": big_3 if len(big_3) == 3 else big_3 + ["待更新"] * (3 - len(big_3)),
            "recommendations": recommendations if len(recommendations) == 3 else recommendations + ["待更新"] * (3 - len(recommendations))
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})


# **下載心情趨勢圖**
@app.route("/get_mood_trend/<user_id>", methods=["GET"])
def get_mood_trend(user_id):
    file_path = f"static/moodtrend/mood_trend_{user_id}.png"
    if os.path.exists(file_path):
        return send_file(file_path, mimetype="image/png")
    return jsonify({"error": "圖片不存在"}), 404


if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # ✅ 修正 Windows 的 asyncio 問題
    app.run(debug=True, use_reloader=False)
