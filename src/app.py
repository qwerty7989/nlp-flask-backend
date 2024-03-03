from flask import Flask, render_template, request
from nlp import nlp

app = Flask(__name__)

@app.route("/")
def index():
        return render_template('index.html')

@app.route("/result")
def result():
        text = request.args.get('text')
        nlp_runtime = nlp(text)
        percentage = nlp_runtime.result
        return f'You enter {text} and it is probably {percentage}'
