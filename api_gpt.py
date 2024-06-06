from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import numpy as np

# .envファイルから環境変数を読み込みます
load_dotenv()

# OpenAIのAPIキーを取得します
api_key = os.getenv('OPENAI_APIKEY')

client = OpenAI(
    api_key=api_key,
)

# プロンプトを設定します
prompt_template = """
Summarize the Disney work in 5 sentences according to the following structure:
1. Beginning
2. Trigger 1
3. Middle
4. Trigger 2
5. End

Output in array format ([sentence1,sentence2,sentence3,sentence4,sentence5]). The output should be in English. 

Summarize the following Disney work: {}
"""

# 30作品のタイトルをリストにします
disney_works = [
    "The Lion King", "Frozen", "Beauty and the Beast", "Aladdin", "Mulan",
    "The Little Mermaid", "Pocahontas", "Hercules", "Tarzan", "Cinderella",
    "Sleeping Beauty", "Snow White and the Seven Dwarfs", "Tangled", "Moana",
    "Zootopia", "Big Hero 6", "Wreck-It Ralph", "The Incredibles", "Finding Nemo",
    "Toy Story", "Monsters, Inc.", "Cars", "Brave", "Ratatouille", "Up",
    "Inside Out", "Coco", "Onward", "Soul", "Luca"
]

# 返答を取得する関数
def get_summary(work):
    prompt = prompt_template.format(work)
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }],
        model="gpt-4o",
    )
    return response.choices[0].message.content
# 30作品の要約を取得します
summaries = {}
for work in disney_works:
    summaries[work] = get_summary(work)



# 結果をnpy形式で保存します
np.save('smp_data.npy', summaries)

# 結果を表示します
for work, summary in summaries.items():
    print(f"{work}: {summary}")
