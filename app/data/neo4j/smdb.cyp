/*'expertType', 'degree', 'variation', 'universityCategory', 'midinspectionAuditOpinion', 'newsType', 'indexType',
  'disciplineType', 'idcardType', 'GBT2260-2007', 'specialityTitle', 'tutorType', 'awardReviewIndex', 'businessType',
  'ISO3166-1', 'GBT4881-1985', 'projectFundingType', 'productType', 'projectType', 'researchActivityType',
  'GBT13745-2009', 'GBT8561-2001', 'workFundingType', 'talentType', 'GB3304-91', 'universityStyle',
  'endinspectionAuditOpinion', 'ISO 639-1', 'bank', 'membership', 'researchAgencyType'
*/

WITH {
  expertType: '专家类型', degree: '学位', variation: '变更事项', universityCategory: '高校性质类别',
  midinspectionAuditOpinion: '中检审核意见', newsType: '新闻类别', indexType: '索引类型', disciplineType: '学科门类',
  idcardType: '证件类型', gbt2260: '中华人民共和国行政区划代码', specialityTitle: '专业职称', tutorType: '导师类型',
  awardReviewIndex: '奖励评审指标', businessType: '业务类型', iso3166: '国家和地区代码', gbt4881: '少数民族语言',
  projectFundingType: '项目拨款类型', productType: '成果形式', projectType: '项目类型', researchActivityType: '研究活动类型',
  gbt13745: '学科分类与代码表', gbt8561: '专业技术职务', workFundingType: '工作拨款类型', talentType: '人才类型',
  gb3304: '中国各民族名称的罗马字母拼写法和代码', universityStyle: '高校办学类型', endinspectionAuditOpinion: '结项审核意见',
  iso639: '外语语种', bank: '银行', membership: '政治面貌', researchAgencyType: '研究机构类型'
} AS kv

// load option data
USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///option.csv' AS cl
CREATE (option:Option {id: cl.id, name: cl.name, standard: cl.standard})

//set up the relationship of option data
USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///option.csv' AS cl
MATCH (option:Option {id: cl.id}), (parent:Option {id: cl.parent_id})
CREATE (option)-[:CHILD_OF]->(parent)


// load agency data
USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///agency.csv' AS cl
CREATE (agency:Agency {id: cl.id, name: cl.name, type: cl.type, style: cl.style, category: cl.category,
        organizer: cl.organizer, standard: cl.standard, istop: cl.istop, ismiddle: cl.ismiddle, homepage: cl.homepage})

//set up the relationship of agency data
USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///agency.csv' AS cl
MATCH (agency:Agency {id: cl.id}), (parent:Agency {id: cl.subjection_id}), (province:Option {id: cl.province_id})
CREATE (agency)-[:CHILD_OF]->(parent), (agency)-[:IN_PLACE]->(province)


// load person data
USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///person.csv' AS cl
CREATE (person:Person {id: cl.id, name: cl.name, gender: cl.gender, email: cl.email, phone: cl.phone,
        language: cl.language, research_field: cl.research_field, degree: cl.degree})

//set up the relationship of person data
USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///person.csv' AS cl
MATCH (person:Person {id: cl.id}), (discipline:Option {name: cl.discipline, standard: 'GBT13745-2009'})
CREATE (person)-[:MAJOR_IN]->(discipline)


// load application data
USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///application.csv' AS cl
CREATE (app:Application {id: cl.id, name: cl.name, type: cl.type, product_type: cl.product_type, year: cl.year,
        discipline_type: cl.discipline_type, plan_end_date: cl.plan_end_date, status: cl.status,
        applicant_submit_date: cl.applicant_submit_date, final_audit_date: cl.final_audit_date})

//set up the relationship of application data
/*foreach (applicant_id in split(cl.applicant_id, "; ")| match (applicant:Person {id: applicant_id})
                                create (app)-[:APPLICANT {name: applicant.name}]->(applicant))*/
USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///application.csv' AS cl
MATCH (app:Application {id: cl.id}), (subtype:Option {id: cl.subtype_id}),
      (discipline:Option {name: cl.discipline, standard: 'GBT13745-2009'}),
      (restype:Option {id: cl.research_type_id}), (university:Agency {id: cl.university_id}),
      (department:Option {id: cl.department_id}), (auditor:Person {id: cl.final_auditor_id}),
      (auditor_agency:Agency {id: cl.final_auditor_agency_id}), (province:Option {id: cl.province_id}),
      (applicant:Person {id: cl.applicant_id})
CREATE (app)-[:SUB_TYPE]->(subtype), (app)-[:DISCIPLINE]->(discipline), (app)-[:RESEARCH_TYPE]->(restype),
       (app)-[:UNIVERSITY]->(university), (app)-[:DEPARTMENT]->(department), (app)-[:AUDITOR]->(auditor),
       (app)-[:AUDITOR_AGENCY]->(auditor_agency), (app)-[:PROVINCE]->(province), (app)-[:APPLICANT]->(applicant)


// load discipline data
USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///discipline.csv' AS cl
CREATE (discipline:Discipline {id: cl.id, name: cl.name, code: cl.code})

//set up the relationship of discipline data
USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///discipline.csv' AS cl
MATCH (discipline:Discipline {id: cl.id}), (d:Option {name: cl.discipline}), (university:Agency {id: cl.university_id})
CREATE (discipline)-[:TYPE]->(d), (discipline)-[:BELONG_TO]->(university)


// load doctoral data
USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///doctoral.csv' AS cl
CREATE (doctoral:Doctoral {id: cl.id, name: cl.name, code: cl.code, is_key: cl.is_key})

//set up the relationship of doctoral data
USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///doctoral.csv' AS cl
MATCH (doctoral:Doctoral {id: cl.id}), (d:Option {name: cl.discipline}), (university:Agency {id: cl.university_id})
CREATE (doctoral)-[:TYPE]->(d), (doctoral)-[:BELONG_TO]->(university)


// create unique constraint
CREATE CONSTRAINT ON (opt:Option) ASSERT opt.id IS UNIQUE
CREATE CONSTRAINT ON (age:Agency) ASSERT age.id IS UNIQUE
CREATE CONSTRAINT ON (per:Person) ASSERT per.id IS UNIQUE
CREATE CONSTRAINT ON (app:Application) ASSERT app.id IS UNIQUE
CREATE CONSTRAINT ON (dis:Discipline) ASSERT dis.id IS UNIQUE
CREATE CONSTRAINT ON (doc:Doctoral) ASSERT doc.id IS UNIQUE