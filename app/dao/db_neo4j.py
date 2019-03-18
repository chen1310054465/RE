import json
import os
import random
import yaml

from neo4j import GraphDatabase
from flask import g

pDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
p2attr = {"applicant_submit_date": "申请时间", "category": "性质类别", "code": "代码", "discipline_type": "学科类型",
          "email": "电子邮件", "final_audit_date": "终审时间", "gender": "性别", "homepage": "主页", "id": "唯一标识",
          "is_key": "属于", "ismiddle": "属于", "istop": "属于", "language": "语言", "name": "名称",
          "organizer": "高校举办者", "phone": "电话", "plan_end_date": "项目计划完成时间", "product_type": "成果类型",
          "research_field": "研究领域", "standard": "标准", "status": "状态", "style": "高校办学类型", "type": "类型",
          "year": "申请年份"}
agency_type = {"1": "部级", "2": "省级", "3": "部属高校", "4": "地方高校"}
app_type = {"instp": "基地项目", "devrpt": "发展报告项目", "entrust": "委托应急项目", "general": "一般项目",
            "post": "后期资助项目", "special": "专项项目", "key": "重大攻关项目"}
app_status = {"0": "默认", "1": "新建申请", "2": "院系/研究机构审核", "3": "校级审核", "4": "省级审核",
              "5": "部级审核", "6": "评审", "7": "评审结果审核", "8": "最终审核"}


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
    for record in tx.run("MATCH (n {name: $name}) "
                         "RETURN n", name=name):
        label = list(record['n'].labels)[0]
        for k in record['n'].keys():
            if k not in ['id', 'name']:
                value = "重点" if k == "is_key" else "211高校" if k == "ismiddle" else "985高校" if k == "istop" \
                    else agency_type[record['n'][k]] if label == "Agency" and k == "type" else record['n'][k]
                if label == "Application":
                    value = app_type[record['n'][k]] if k == "type" else app_status[record['n'][k]] if k == "status" \
                        else value
                result["relations"].append({"attr": p2attr[k], "value": value, "entity": value,
                                            "click": random.randrange(1e4, 1e6), "pic": "", "desc": p2attr[k]})

    for record in tx.run("MATCH (s)-[p]-(o) "
                         "WHERE s.name = {name} "
                         "RETURN s, p, o", name=name):
        sub_name = "<a href=\"" + record["s"]['name'] + "\">" + record["s"]['name'] + "</a>"
        obj_name = "<a href=\"" + record["o"]['name'] + "\">" + record["o"]['name'] + "</a>"
        result["property"]["related"].append({"o": obj_name, "context": sub_name, "click": random.randrange(1e4, 1e6),
                                              "desc": record["o"]['name']})

    # print(sub_name, "--", record["p"].type, "--", obj_name)
    return json.dumps(result, ensure_ascii=False)


with open(pDir + '/resources/app.yml', 'r') as f:
    conf = yaml.load(f)
d = get_driver(conf['neo4j_url'], conf['neo4j_username'], conf['neo4j_password'])


def get_graph_data(entity):
    with d.session() as sess:
        res = sess.read_transaction(get_info_of_entity, entity)
        return res


if __name__ == '__main__':
    print(get_graph_data("数学"))
