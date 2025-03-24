from flask import Flask, request, jsonify, render_template
import pandas as pd
import os

app = Flask(__name__)

# 確保 moodtrend 資料夾存在
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'moodtrend')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 存儲分析結果（假設是用戶 1）
analysis_result = {
    "user_id": 1,
    "big_3_goals": ["多運動", "早睡早起", "每天寫日記"],
    "recommendations": ["多進行戶外活動", "保持規律作息", "與朋友多聊天"],
    "mood_trend": [5, 6, 4, 7, 8, 6]  # 假設的心情指數
}

@app.route('/')
def index():
    return render_template("indexv2.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    """ 處理 CSV 檔案上傳 """
    if 'file' not in request.files:
        return jsonify({"error": "沒有選擇檔案"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "檔案名稱為空"}), 400
    
    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # 進行心情分析
        df = pd.read_csv(filepath)
        process_mood_analysis(df)  # 分析心情趨勢
        return jsonify({"message": "檔案上傳成功並已分析"})
    
    return jsonify({"error": "請上傳 CSV 檔案"}), 400

@app.route('/get_recommendations', methods=['GET'])
def get_recommendations():
    """ 提供 Big 3 目標與建議 """
    return jsonify(analysis_result)

def process_mood_analysis(df):
    """ 假設這裡分析 CSV 內容，根據使用者數據設定 Big 3 目標與建議 """
    global analysis_result

    # 檢查並處理列名，確保正確使用 "心情指數"
    if "心情指數" not in df.columns:
        raise ValueError("CSV 檔案缺少 '心情指數' 列")
    
    # 如果需要，可以將列名 "心情指數" 改為 "Mood" 以便後續處理
    df.rename(columns={"心情指數": "Mood"}, inplace=True)

    user_id = 1  # 這裡假設是用戶 1，實際上可根據 CSV 動態決定
    avg_mood = df["Mood"].mean()

    # 根據心情指數來設定 Big 3 目標與建議
    if avg_mood < 5:
        big_3 = ["每天多笑笑", "多參與社交活動", "設定小目標"]
        recommendations = ["每天試著微笑 10 次", "多與朋友聯繫", "培養新興趣"]
    elif avg_mood < 7:
        big_3 = ["規律作息", "每日運動", "減少社群媒體使用"]
        recommendations = ["保持固定睡眠時間", "每天運動 30 分鐘", "減少滑手機時間"]
    else:
        big_3 = ["繼續保持好習慣", "挑戰新事物", "幫助他人"]
        recommendations = ["參與志工活動", "嘗試新技能", "與家人分享快樂時光"]

    # 更新分析結果
    analysis_result = {
        "user_id": user_id,
        "big_3_goals": big_3,
        "recommendations": recommendations,
        "mood_trend": df["Mood"].tolist()
    }

@app.route("/result")
def result():
    # 假設你已經計算好了分析結果
    big_3_goals = analysis_result["big_3_goals"]
    recommendations = analysis_result["recommendations"]

    return render_template("indexv2.html", big_3_goals=big_3_goals, recommendations=recommendations)


if __name__ == '__main__':
    app.run(debug=True)
