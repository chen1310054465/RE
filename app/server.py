from flask import Flask, request
import requests

app = Flask(__name__)


@app.route("/")
def index():
    r = requests.get('http://shuyantech.com/cndbpedia/kggraph?' + str(request.query_string, encoding='utf-8'))
    # return r.text.replace('/semantic/dist/', 'http://shuyantech.com/semantic/dist/') \
    #     .replace('/css/', 'http://shuyantech.com/css/') \
    #     .replace('/scripts/', 'http://shuyantech.com/scripts/')
    return r.text.replace('/semantic/dist/', '/static/semantic/dist/') \
        .replace('/css/', '/static/css/') \
        .replace('/scripts/', '/static/scripts/')


@app.route("/cndbpedia/kggraphData")
def kg_graph_data():
    r = requests.get('http://shuyantech.com/cndbpedia/kggraphData?' + str(request.query_string, encoding='utf-8'))
    return r.text


@app.route("/cndbpedia/kggraphConcepts")
def kg_graph_concepts():
    r = requests.get('http://shuyantech.com/cndbpedia/kggraphConcepts?' + str(request.query_string, encoding='utf-8'))
    return r.text


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)
