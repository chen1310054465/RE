import re

import requests
from flask import Flask, request, render_template

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/kgView")
def kg_view():
    return render_template('kg_view.html')


@app.route("/kgv")
def kgv():
    r = requests.get('http://shuyantech.com/cndbpedia/kggraph?' + str(request.query_string, encoding='utf-8'))
    text = r.text
    # text = re.sub('/(css|semantic/dist|scripts)/(.*?)\.(css|js)', 'http://shuyantech.com/\g<1>/\g<2>.\g<3>', text)
    text = re.sub('/(css|semantic/dist)/(.*?)\.css', '/static/css/\g<2>.css', text)
    text = re.sub('/(scripts|semantic/dist)/(.*?)\.js', '/static/js/\g<2>.js', text)
    text = re.sub('>KGGraph<', '>KG Visualization<', text)
    text = re.sub('velocity\.min\.js"></script>',
                  'velocity.min.js"></script>\n  <link rel="shortcut icon" href="/static/img/favicon.ico">', text)
    return text


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
