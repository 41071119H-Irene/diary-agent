import os
import asyncio
import json
import threading
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from google import genai
from EMOwithSnow import generate_mood_trend_plot
from multiagent import run_multiagent_analysis

# ✅ 初始化 Flask 與 SocketIO
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
socketio = SocketIO(app, async_mode='threading')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ✅ 載入 .env 並初始化預設 Gemini API KEY
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ✅ 預設 Gemini client（用於即時對話）
client = genai.Client(api_key=GEMINI_API_KEY)

# ✅ 首頁路由
@app.route('/')
def index():
    return render_template('index.html')

# ✅ 檔案上傳路由
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    is_counselor = request.form.get('is_counselor', 'false').lower() == 'true'
    api_key = request.form.get('api_key', '').strip()

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        socketio.emit('update', {'message': '🟢 檔案上傳成功，開始分析中...'})

        # ✅ 執行背景任務，傳入 api_key
        threading.Thread(target=background_task, args=(file_path, is_counselor, api_key)).start()
        return 'File uploaded and processing started.', 200

# ✅ 背景分析任務
def background_task(file_path, is_counselor, api_key):
    try:
        df = pd.read_csv(file_path)
        user_id = os.path.splitext(os.path.basename(file_path))[0]
        plot_path = generate_mood_trend_plot(user_id, df)
        socketio.emit('plot_generated', {'plot_url': '/' + plot_path})
        print(f"[DEBUG] 收到的 API KEY：{api_key[:8]}...（共 {len(api_key)} 字元）")

        # ✅ 如果使用者有輸入 API KEY，用該 key 建立 client
        if api_key:
            try:
                custom_client = genai.Client(api_key=api_key)
                from mcp import ModelClient
                ModelClient.default_client = custom_client
            except Exception as e:
                socketio.emit('update', {'message': f"❌ API KEY 驗證失敗：{str(e)}"})
                return

        asyncio.run(run_multiagent_analysis(socketio, user_id, df, is_counselor))

    except Exception as e:
        socketio.emit('update', {'message': f"❌ 分析過程出現錯誤：{str(e)}"})

# ✅ 即時聊天室對話功能
@socketio.on('chat_message')
def handle_user_chat(data):
    user_message = data.get('message', '').strip()
    if not user_message:
        return

    socketio.emit('ai_reply', {'message': '💬 Gemini 正在思考中，請稍候...'})

    def chat_reply():
        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=f"你是心理諮商師，要用正體中文回應以下內容：{user_message}"
            )
            reply = response.text.strip()
            socketio.emit('ai_reply', {'message': reply})
        except Exception as e:
            socketio.emit('ai_reply', {'message': f"⚠️ 發生錯誤：{str(e)}"})

    threading.Thread(target=chat_reply).start()

if __name__ == '__main__':
    socketio.run(app, debug=True)
