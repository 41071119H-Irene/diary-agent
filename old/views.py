from django.shortcuts import render 
from blog.models import TeachingSession
from myproject.utils import ask_chatgpt
import openai
import os
import hashlib
import hmac
from django.conf import settings

# 設定 OpenAI API Key
openai.api_key = settings.OPENAI_API_KEY

# 生成 Chatbase 驗證哈希
def generate_chatbase_hash(user_id):
    secret_key = settings.CHATBASE_SECRET_KEY.encode('utf-8')
    user_id_bytes = str(user_id).encode('utf-8')
    return hmac.new(secret_key, user_id_bytes, hashlib.sha256).hexdigest()

# 主要聊天視圖
def chat_view(request):
    if request.method == 'GET':
        request.session.flush()  # 清除對話數據
        user_id = request.session.session_key

        if not user_id:
            request.session.create()
            user_id = request.session.session_key

        # 取得或建立 TeachingSession
        session, _ = TeachingSession.objects.get_or_create(user_id=user_id)
        session.current_step = 0  
        session.save()

        # ChatGPT 先主動問候使用者
        chatgpt_intro = "您好，我是小花，請問有什麼問題嗎？"
        conversation = [{"sender": "ai", "message": chatgpt_intro, "source": "system"}]

        # 產生 Chatbase 驗證哈希
        chatbase_hash = generate_chatbase_hash(user_id)

        request.session['conversation'] = conversation
        return render(request, 'blog/chat_window.html', {
            'conversation': conversation,
            'chatbase_hash': chatbase_hash,
            'user_id': user_id
        })

    if request.method == 'POST':
        conversation = request.session.get('conversation', [])
        user_message = request.POST.get('message', '').strip()
        
        if user_message:
            conversation.append({"sender": "user", "message": user_message})

            # ChatGPT 根據設定的知識庫回答
            chatgpt_response = ask_chatgpt(user_message)
            conversation.append({"sender": "ai", "message": chatgpt_response, "source": "chatgpt"})

        request.session['conversation'] = conversation
        return render(request, 'blog/chat_window.html', {
            'conversation': conversation,
            'chatbase_hash': generate_chatbase_hash(request.session.session_key),
            'user_id': request.session.session_key
        })
