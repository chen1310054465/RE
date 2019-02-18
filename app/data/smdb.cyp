/*'expertType', 'degree', 'variation', 'universityCategory', 'midinspectionAuditOpinion', 'newsType', 'indexType',
  'disciplineType', 'idcardType', 'GBT2260-2007', 'specialityTitle', 'tutorType', 'awardReviewIndex', 'businessType',
  'ISO3166-1', 'GBT4881-1985', 'projectFundingType', 'productType', 'projectType', 'researchActivityType',
  'GBT13745-2009', 'GBT8561-2001', 'workFundingType', 'talentType', 'GB3304-91', 'universityStyle',
  'endinspectionAuditOpinion', 'ISO 639-1', 'bank', 'membership', 'researchAgencyType'
*/

with {
  expertType: "专家类型", degree: "学位", variation: "变更事项", universityCategory: "高校性质类别",
  midinspectionAuditOpinion: "中检审核意见", newsType: "新闻类别", indexType: "索引类型", disciplineType: "学科门类",
  idcardType: "证件类型", gbt2260: "中华人民共和国行政区划代码", specialityTitle: "专业职称", tutorType: "导师类型",
  awardReviewIndex: "奖励评审指标", businessType: "业务类型", iso3166: "国家和地区代码", gbt4881: "少数民族语言",
  projectFundingType: "项目拨款类型", productType: "成果形式", projectType: "项目类型", researchActivityType: "研究活动类型",
  gbt13745: "学科分类与代码表", gbt8561: "专业技术职务", workFundingType: "工作拨款类型", talentType: "人才类型",
  gb3304: "中国各民族名称的罗马字母拼写法和代码", universityStyle: "高校办学类型", endinspectionAuditOpinion: "结项审核意见",
  iso639: "外语语种", bank: "银行", membership: "政治面貌", researchAgencyType: "研究机构类型"
} as kv

// load option data
using periodic commit 500
load csv with headers from 'file:///option.csv' as cl
create (option:Option {id: cl.id, name: cl.name, standard: cl.standard})
return option

//set up the relationship of option data
using periodic commit 500
load csv with headers from 'file:///option.csv' as cl
match (option:Option {id: cl.id}), (parent:Option {id: cl.parent_id})
create (option)-[:CHILD_OF]->(parent)
return option, parent


// load agency data
using periodic commit 500
load csv with headers from 'file:///agency.csv' as cl
create (agency:Agency {id: cl.id, name: cl.name, type: cl.type, style: cl.style, category: cl.category,
        organizer: cl.organizer, standard: cl.standard, istop: cl.istop, ismiddle: cl.ismiddle, homepage: cl.homepage})
return agency

//set up the relationship of agency data
using periodic commit 500
load csv with headers from 'file:///agency.csv' as cl
match (agency:Agency {id: cl.id}), (parent:Agency {id: cl.subjection_id}), (province:Option {id: cl.province_id})
create (agency)-[:CHILD_OF]->(parent), (agency)-[:IN_PLACE]->(province)
return agency, parent


// load person data
using periodic commit 500
load csv with headers from 'file:///person.csv' as cl
create (person:Person {id: cl.id, name: cl.name, gender: cl.gender, email: cl.email, phone: cl.phone,
        language: cl.language, research_field: cl.research_field, degree: cl.degree})
return person

//set up the relationship of person data
using periodic commit 500
load csv with headers from 'file:///person.csv' as cl
match (person:Person {id: cl.id}), (discipline:Option {name: cl.discipline})
create (person)-[:MAJOR_IN]->(discipline)
return person


// load application data
using periodic commit 500
load csv with headers from 'file:///application.csv' as cl
create (app:Application {id: cl.id, name: cl.name, type: cl.type, product_type: cl.product_type, year: cl.year,
        discipline_type: cl.discipline_type, plan_end_date: cl.plan_end_date, status: cl.status,
        applicant_submit_date: cl.applicant_submit_date, final_audit_date: cl.final_audit_date})
return app

//set up the relationship of application data
using periodic commit 500
load csv with headers from 'file:///application.csv' as cl
match (app:Application {id: cl.id}), (subtype:Option {id: cl.subtype_id}), (discipline:Option {name: cl.discipline}),
      (restype:Option {id: cl.research_type_id}), (applicant:Person {id: cl.applicant_id}),
      (university:Agency {id: cl.university_id}), (department:Option {id: cl.department_id}),
      (auditor:Person {id: cl.final_auditor_id}), (auditor_agency:Agency {id: cl.final_auditor_agency_id}),
      (province:Option {id: cl.province_id})
create (app)-[:SUB_TYPE]->(subtype), (app)-[:DISCIPLINE]->(discipline), (app)-[:RESEARCH_TYPE]->(restype),
       (app)-[:APPLICANT]->(applicant), (app)-[:UNIVERSITY]->(university), (app)-[:DEPARTMENT]->(department),
       (app)-[:AUDITOR]->(auditor), (app)-[:AUDITOR_AGENCY]->(auditor_agency), (app)-[:PROVINCE]->(province)
return app


// load discipline data
using periodic commit 500
load csv with headers from 'file:///discipline.csv' as cl
create (discipline:Discipline {id: cl.id, name: cl.name, code: cl.code})
return discipline

//set up the relationship of discipline data
using periodic commit 500
load csv with headers from 'file:///discipline.csv' as cl
match (discipline:Discipline {id: cl.id}), (d:Option {name: cl.discipline}), (university:Agency {id: cl.university_id})
create (discipline)-[:TYPE]->(d), (discipline)-[:BELONG_TO]->(university)
return discipline, d, university


// load doctoral data
using periodic commit 500
load csv with headers from 'file:///doctoral.csv' as cl
create (doctoral:Doctoral {id: cl.id, name: cl.name, code: cl.code, is_key: cl.is_key})
return doctoral

//set up the relationship of doctoral data
using periodic commit 500
load csv with headers from 'file:///doctoral.csv' as cl
match (doctoral:Doctoral {id: cl.id}), (d:Option {name: cl.discipline}), (university:Agency {id: cl.university_id})
create (doctoral)-[:TYPE]->(d), (doctoral)-[:BELONG_TO]->(university)
return doctoral, d, university