import json

from neo4j import GraphDatabase
from flask import g


def get_driver(uri, username, password):
    return GraphDatabase.driver(uri, auth=(username, password))


def get_sess(driver):
    if not hasattr(g, 'neo4j_sess'):
        g.neo4j_sess = driver.session()
    return g.neo4j_sess


# @app.teardown_appcontext
def close_sess():
    if hasattr(g, 'neo4j_sess'):
        g.neo4j_sess.close()


def get_info_of_entity(tx, name):
    result = {"status": "ok", "property": {"name": name, "pic": "", "desc": name, "related": []}, "relations": []}
    for record in tx.run("MATCH (s)-[p]-(o) "
                         "WHERE s.name = {name} "
                         "RETURN s, p, o", name=name):
        result["property"]["related"].append({"o": record["o"]['name'], "context": record["o"]['name'], "click": 1000,
                                              "desc": record["o"]['name']})

        print(record["s"]['name'], "--", record["p"].type, "--", record["o"]['name'])
    return json.dumps(result, ensure_ascii=False)


d = get_driver("bolt://192.168.88.23:7687", "neo4j", "zhaohq5133")


def get_graph_data(entity):
    with d.session() as sess:
        res = sess.read_transaction(get_info_of_entity, entity)
        return res


if __name__ == '__main__':
    print(get_graph_data("数学"))
