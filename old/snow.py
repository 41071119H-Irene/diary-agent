import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from snownlp import SnowNLP

matplotlib.rc('font', family='Microsoft JhengHei')

# 設定 CSV 檔案路徑
csv_file_path = "user_diary.csv"  # 請修改為你的檔案路徑

# 檢查檔案是否存在
if not os.path.exists(csv_file_path):
    print(f"❌ 找不到檔案 {csv_file_path}，請確認檔案是否存在！")
    exit()

# 讀取 CSV
df = pd.read_csv(csv_file_path)

# 確保日期格式正確
df["日期"] = pd.to_datetime(df["日期"])
df = df.sort_values("日期")

# 確保心情指數為數字
df["心情指數"] = pd.to_numeric(df["心情指數"], errors="coerce")

# 用 SnowNLP 進行情感分析，並命名為心情小語分析
df["心情小語分析"] = df["心情小語"].apply(lambda text: SnowNLP(text).sentiments * 9 + 1)

# 設定目標用戶 ID
target_user_id = 10  # 請修改成你要的 ID

# 只篩選特定用戶
if target_user_id in df["用戶ID"].unique():
    user_entries = df[df["用戶ID"] == target_user_id]

    plt.figure(figsize=(10, 5))

    # 繪製每日心情指數
    sns.lineplot(x=user_entries["日期"], y=user_entries["心情指數"], marker="o", label="心情指數", color="blue", ci=None)
    sns.lineplot(x=user_entries["日期"], y=user_entries["心情小語分析"], marker="o", label="心情小語分析", color="red", ci=None)

    # 計算並繪製平均心情指數
    avg_mood = user_entries["心情指數"].mean()
    plt.axhline(y=avg_mood, color='orange', linestyle='--', label=f"平均心情指數 ({avg_mood:.2f})")

    # 設定標題與標籤
    plt.xlabel("日期")
    plt.ylabel("心情指數")
    plt.title(f"用戶 {target_user_id} 的心情趨勢")
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.ylim(1, 10)  # 調整 Y 軸範圍與表格一致

    # ✅ 顯示圖表
    plt.show()
else:
    print(f"❌ 找不到用戶 ID {target_user_id}，請確認數據是否正確！")
