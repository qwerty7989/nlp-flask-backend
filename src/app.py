from flask import Flask, render_template, request
from nlp import calculate_value

app = Flask(__name__)

@app.route("/")
def index():
        return render_template('index.html')

@app.route("/result")
def result():
        text = request.args.get('text')
        percentage = calculate_value(text)
        return f'You enter {text} and it is probably {percentage}'
