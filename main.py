from flask import Flask, render_template, request
from src.response import generateResponse

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chat():
    userQuery = request.form["msg"]
    result = generateResponse(userQuery)
    return result

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)