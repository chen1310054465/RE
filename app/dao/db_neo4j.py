import json
import random

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
    p2attr = {"applicant_submit_date": "申请时间", "category": "性质类别", "code": "代码", "discipline_type": "学科类型",
              "email": "电子邮件", "final_audit_date": "终审时间", "gender": "性别", "homepage": "主页", "id": "唯一标识",
              "is_key": "属于", "ismiddle": "属于", "istop": "属于", "language": "语言", "name": "名称",
              "organizer": "高校举办者", "phone": "电话", "plan_end_date": "项目计划完成时间", "product_type": "成果类型",
              "research_field": "研究领域", "standard": "标准", "status": "状态", "style": "高校办学类型", "type": "类型",
              "year": "申请年份"}
    for record in tx.run("MATCH (n {name: $name}) "
                         "RETURN n", name=name):
        # print(record['n'])
        # print(list(record['n'].labels)[0])
        # print(record['n']['standard'])
        for k in record['n'].keys():
            if k not in ['id', 'name']:
                value = "重点" if k == "is_key" else "211高校" if k == "ismiddle" else "985高校" if k == "istop" \
                    else record['n'][k]
                result["relations"].append({"attr": p2attr[k], "value": value, "entity": value,
                                            "click": random.randrange(1e3, 1e6), "pic": "", "desc": p2attr[k]})

    for record in tx.run("MATCH (s)-[p]-(o) "
                         "WHERE s.name = {name} "
                         "RETURN s, p, o", name=name):
        sub_name = "<a href=\"" + record["s"]['name'] + "\">" + record["s"]['name'] + "</a>"
        obj_name = "<a href=\"" + record["o"]['name'] + "\">" + record["o"]['name'] + "</a>"
        result["property"]["related"].append({"o": obj_name, "context": sub_name, "click": random.randrange(1e3, 1e6),
                                              "desc": record["o"]['name']})

    # print(sub_name, "--", record["p"].type, "--", obj_name)
    return json.dumps(result, ensure_ascii=False)


d = get_driver("bolt://192.168.88.23:7687", "neo4j", "zhaohq5133")


def get_graph_data(entity):
    with d.session() as sess:
        res = sess.read_transaction(get_info_of_entity, entity)
        return res


if __name__ == '__main__':
    print(get_graph_data("数学"))
