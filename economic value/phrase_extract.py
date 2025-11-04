import pandas as pd
import re

data_csv = pd.read_excel('/content/sample_data/work order history_UCSF_2005-2022 (5) (1).xlsx')

print("len of actual dataset", len(data_csv))

data_csv['WO_LaborReport'] = data_csv['WO_LaborReport'].astype(str).fillna("")

keywords = ["replace","replaced","remove", "removed", "failure"]
keyword_set = set(keywords)

def extract_context_phrases(text):
  words = re.findall(r'\b\w+\b', text.lower())
  phrases = []
  for i, word in enumerate(words):
    if word in keyword_set:
      before = words[max(0,i-3):i]
      if word == "replaced":
        after_words = words[i+1:i+10]
        if after_words:
          phrase = f"{' '.join(before)} ###<{word}###{' '.join(after_words)}"
          phrases.append(phrase)
      else:
        after = words[i+1:i+3]
        phrase = f"{' '.join(before)} ###<{word}###{' '.join(after)}".strip()
        phrases.append(phrase)

  return ' | '.join(phrases) if phrases else ''


data_csv['Extracted_Phrases'] = data_csv['WO_LaborReport'].apply(extract_context_phrases)
filtered_with_phrases = data_csv[data_csv['Extracted_Phrases'] != ""]
only_phrases = filtered_with_phrases[['Extracted_Phrases']]
only_phrases.to_csv("context_phrases_extracted.csv", index=False)

print("Saved extracted phrases to context_phrases_extracted.csv")





