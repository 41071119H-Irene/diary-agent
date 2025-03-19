from flask import Flask, render_template, request, jsonify
import os
import asyncio
from dataAgent import process_chunk, OpenAIChatCompletionClient
from AgentEMO import analyze_sentiment
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_diary', methods=['POST'])
def upload_diary():
    file = request.files['file']
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        # 開始處理日記資料
        df = pd.read_csv(file_path)
        total_records = len(df)
        
        # 假設已經有 OpenAI API client 的設定
        model_client = OpenAIChatCompletionClient(model="gemini-2.0-flash", api_key=os.getenv('GEMINI_API_KEY'))
        
        results = []
        for start_idx in range(0, total_records, 500):
            chunk = df[start_idx:start_idx+500]
            results.extend(asyncio.run(process_chunk(chunk, start_idx, total_records, model_client, "terminate")))
        
        # 送回分析結果
        return jsonify({"status": "success", "results": results})
    return jsonify({"status": "error", "message": "No file uploaded."})

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_user_sentiment():
    text = request.json.get('text')
    if text:
        sentiment_score = analyze_sentiment(text)
        return jsonify({"status": "success", "sentiment_score": sentiment_score})
    return jsonify({"status": "error", "message": "No text provided."})

if __name__ == '__main__':
    app.run(debug=True)
