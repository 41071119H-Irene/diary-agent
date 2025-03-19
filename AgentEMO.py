import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

matplotlib.rc('font', family='Microsoft JhengHei')

# 讀取 CSV 檔案
data = pd.read_csv("user_diary.csv")

# 檢查是否有必要欄位
if '用戶ID' not in data.columns or '日期' not in data.columns or '心情小語' not in data.columns:
    print("⚠️ 錯誤：CSV 檔案缺少必要欄位（用戶ID、日期、心情小語）。")
else:
    # 解析日期格式
    data['日期'] = pd.to_datetime(data['日期'], errors='coerce')
    data = data.dropna(subset=['日期'])

    # 定義情緒分析函數（1~10 分數）
    def analyze_sentiment(text):
        if pd.isna(text):
            return 5  # 預設為中性
        polarity = TextBlob(text).sentiment.polarity  # 取得情緒極性（-1 ~ 1）
        score = round((polarity + 1) * 9 + 1)  # 轉換為 1~10 分數
        return min(max(score, 1), 10)

    # 計算情緒分數
data['情緒分數'] = data['心情小語'].apply(analyze_sentiment)

    # 針對每位用戶繪製情緒走向圖
unique_users = data['用戶ID'].unique()
for user in unique_users:
        user_data = data[data['用戶ID'] == user].sort_values(by='日期')
        plt.figure(figsize=(10, 5))
        sns.lineplot(x='日期', y='情緒分數', data=user_data, marker='o', label=f'用戶 {user}')
        plt.xticks(rotation=45)
        plt.ylim(1, 10)
        plt.xlabel("日期")
        plt.ylabel("情緒分數 (1-10)")
        plt.title(f"用戶 {user} 的情緒趨勢圖")
        plt.legend()
        plt.grid()
        plt.show()
print("✅ 已成功生成所有用戶的情緒趨勢圖！")
