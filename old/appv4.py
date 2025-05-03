import os
import threading
import asyncio
import pandas as pd
from flask_cors import CORS
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
from old.EMOwithSnowv2 import generate_mood_trend_plot
from old.multiagentv2 import run_multiagent_analysis
from chatbot import main  # ✅ 匯入 Gemini 回應功能

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
socketio = SocketIO(app, async_mode='threading')
CORS(app)

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

# 提供 /chat API
@app.route("/main", methods=["POST"])
def chat():
    data = request.get_json()
    prompt = data.get("message", "")
    if not prompt:
        return jsonify({"error": "No message provided"}), 400

    try:
        response = asyncio.run(main(prompt))
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    socketio.run(app, debug=True)
