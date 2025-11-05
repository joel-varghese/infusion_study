import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score,hamming_loss
import ast

# Load both datasets
df = pd.read_csv('/content/sample_data/final_economic_impact.csv')

df['Matched_Components'] = df['Matched_Components'].apply(ast.literal_eval)

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['Matched_Components'])
X = df[['Asset_Serial', 'WO_WO#', 'Past Maintenance', 'Total Active Time']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

clf = OneVsRestClassifier(RandomForestClassifier(random_state=42))
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

predicted_components = mlb.inverse_transform(y_pred)

f1 = f1_score(y_test, y_pred, average='micro')
hl = hamming_loss(y_test, y_pred)

print(f"F1 Score: {f1:.4f}")
print(f"Hamming Loss: {hl:.4f}")