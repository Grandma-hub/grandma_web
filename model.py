import joblib
from nltk.tokenize import RegexpTokenizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

test = pd.read_csv("test_data.csv")
train = pd.read_csv("train_data.csv")
train_x, train_y = train.Text, train.Toxicity
test_x, test_y = test.Text, test.Toxicity

token = RegexpTokenizer(r'[a-zA-Z0-9]+')
vectorizer = TfidfVectorizer(analyzer='word',stop_words="english", lowercase=False, tokenizer=token.tokenize)
vectorizer = vectorizer.fit(train_x)
train_x = vectorizer.transform(train_x)
test_x = vectorizer.transform(test_x)

model = LogisticRegression()
model.fit(train_x, train_y)
predicted = model.predict(test_x)
accuracy_score = metrics.accuracy_score(predicted, test_y)
print(str('{:04.2f}'.format(accuracy_score * 100)) + '%')

joblib.dump(vectorizer, r"vectorizer.sav")
joblib.dump(model, open(r"./model_hs.sav", 'wb'))
