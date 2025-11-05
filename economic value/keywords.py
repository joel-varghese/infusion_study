import pandas as pd
import re
from keybert import KeyBERT

data_csv = pd.read_excel('/content/sample_data/work order history_UCSF_2005-2022 (5) (1).xlsx')

reports = data_csv['WO_LaborReport'].dropna().astype(str)

full_text = " ".join(reports.tolist())

full_text = re.sub(r'\s+', ' ', full_text)

kw_model = KeyBERT()

keywords = kw_model.extract_keywords(
    full_text,
    keyphrase_ngram_range=(1,3),
    stop_words='english',
    use_maxsum=False,
    use_mmr=False,
    top_n=10000
)

print("Top Extracted Keywords:")
for kw, score in keywords:
  print(f"{kw} ({round(score, 2)})")


df = pd.DataFrame(keywords, columns=["Keyword", "Score"])

df.to_csv("extracted_keywords.csv", index=False)