from flask import Flask,render_template,request
import numpy as np
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

def func(x):
  return x.split()

vec_sent = pickle.load(open("./data/vec_sent.pkl",'rb'))
vec_tags = pickle.load(open("./data/vec_tags.pkl",'rb'))
classifier = pickle.load(open("./data/classifier_basic.pkl",'rb'))
top_tags = np.array(vec_tags.get_feature_names())

def rem_html(data):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', str(data))
    return cleantext

stop_words = set(stopwords.words("english"))
stop_words.remove("not")
stem = SnowballStemmer("english")


def preprocess(title,body):
    title = rem_html(title)
    code = str(re.findall(r'<code>(.*?)</code>', body, flags=re.DOTALL))
    body = re.sub('<code>(.*?)</code>', '', body, flags=re.MULTILINE|re.DOTALL)
    body = rem_html(body.encode('utf-8'))
    title=title.encode('utf-8')

    body = str(title)+" "+str(title)+" "+str(title)+" "+str(body)
    body = re.sub(r'[^A-Za-z]+',' ',body)
    words = word_tokenize(str(body.lower()))

    #Removing all single letter and and stopwords from question exceptt for the letter 'c'
    body = ' '.join(str(stem.stem(j)) for j in words if j not in stop_words and (len(j)!=1 or j=='c'))
    return body


def get_tags(title,body):
  body = preprocess(title,body)
  print(body)
  sent_vec = vec_sent.transform([body])
  prob = classifier.predict_proba(sent_vec)
  lst = [top_tags[idx] for idx,p in enumerate(prob[0]) if p >= 0.3]
  return lst



app = Flask(__name__)

@app.route("/",methods=["GET","POST"])
def index():
    return render_template("index.html")
    

@app.route("/tags",methods=["POST"])
def tags():
    title = request.form.get("title_text")
    body = request.form.get("body_text")
    tags = get_tags(title,body)
    print(tags)
    if(len(tags) == 0): return render_template("index.html")
    return render_template("tags.html",tags=tags)


if __name__ == "__main__":
    app.run(port=1200,debug=True)
