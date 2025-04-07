import os
import asyncio
import pandas as pd
from quart import Quart, render_template, request, jsonify, send_from_directory, make_response
from dotenv import load_dotenv
from old.EMOwithSnowv1 import generate_mood_trend_plot, process_user_diary
from autogen_ext.models.openai import OpenAIChatCompletionClient

# 載入環境變數
load_dotenv()

# Quart 應用程式
app = Quart(__name__)

# 設定靜態檔案夾
UPLOAD_FOLDER = "static/moodtrend"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
CSV_FILE_PATH = "user_diary.csv"

# 初始化 AI 客戶端
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("請設定 GEMINI_API_KEY 環境變數")

model_client = OpenAIChatCompletionClient(
    model="gemini-2.0-flash",
    api_key=gemini_api_key
)

# 🟢 分析用戶日記
async def analyze_user_diary(user_id):
    if not os.path.exists(CSV_FILE_PATH):
        return None, None, None

    df = pd.read_csv(CSV_FILE_PATH)
    df.rename(columns={"用戶ID": "user_id"}, inplace=True)

    if user_id not in df["user_id"].unique():
        return None, None, None

    user_entries = df[df["user_id"] == user_id]
    trend_image_path = generate_mood_trend_plot(user_id, user_entries)

    # 設定終止條件
    termination_condition = None
    messages = await process_user_diary(user_id, user_entries, model_client, termination_condition)

    analysis_results = [msg["content"] for msg in messages[:3]] if messages else ["無數據"] * 3
    recommendations = ["練習正向思考", "多參與社交活動", "建立健康生活習慣"]

    return analysis_results, recommendations, trend_image_path

# 🟢 首頁
@app.route('/')
async def index():
    return await render_template('index.html')

# 🟢 提供靜態圖片
@app.route('/static/moodtrend/<filename>')
async def mood_trend_image(filename):
    return await send_from_directory("static/moodtrend", filename)

# 🟢 上傳新日記並分析
@app.route('/upload', methods=('POST',))
async def process_form_data():
    # 取得所有上傳的文件
    for name, file in (await request.files).items():
        # 輸出文件名稱和大小
        file_data = await file.read()  # 讀取文件的內容
        print(f'Processing file: {name} (Size: {len(file_data)} bytes)')

        # 儲存文件到指定的路徑
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, 'wb') as f:
            f.write(file_data)  # 儲存文件內容

        # 假設 user_id 固定為 1，你可以根據需要修改
        user_id = 1

        # 執行分析
        analysis_results, recommendations, trend_image_path = await analyze_user_diary(user_id)

        if not analysis_results:
            return jsonify({'error': '分析失敗'}), 500

        # 修正圖片 URL
        mood_trend_image_url = f"/static/moodtrend/mood_trend_{user_id}.png"

        return jsonify({
            'analysis': analysis_results,
            'recommendations': recommendations,
            'mood_trend_image': mood_trend_image_url
        })

if __name__ == "__main__":
    app.run(debug=True)
