import os
import threading
import asyncio
import pandas as pd
from flask import Flask, render_template, request
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
from EMOwithSnow import generate_mood_trend_plot
from multiagent import run_multiagent_analysis
from chatbot import generate_chat_response  # ✅ 匯入 Gemini 回應功能

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
socketio = SocketIO(app, async_mode='threading')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def background_task(file_path):
    try:
        df = pd.read_csv(file_path)
        user_id = os.path.splitext(os.path.basename(file_path))[0]
        plot_path = generate_mood_trend_plot(user_id, df)
        socketio.emit('plot_generated', {'plot_url': '/' + plot_path})
        asyncio.run(run_multiagent_analysis(socketio, user_id, df))
    except Exception as e:
        socketio.emit('update', {'message': f"❌ 分析過程出現錯誤: {str(e)}"})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        socketio.emit('update', {'message': '🟢 檔案上傳成功，開始分析中...'})
        thread = threading.Thread(target=background_task, args=(file_path,))
        thread.start()
        return 'File uploaded and processing started.', 200

# 🔸 AI 對話功能
@socketio.on('user_message')
def handle_user_message(data):
    user_msg = data['message']
    response = generate_chat_response(user_msg)  # ✅ 呼叫 Gemini 模型取得回應
    socketio.emit('bot_reply', {'message': f"🤖 機器人回覆：{response}"})

if __name__ == '__main__':
    socketio.run(app, debug=True)
