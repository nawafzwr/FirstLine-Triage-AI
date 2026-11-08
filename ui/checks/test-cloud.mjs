import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {createLocalDB} from './d1-local.mjs';
import {createPredictor} from '../worker/model-runtime.js';
import {validateMapping,uncertaintyGuard} from '../worker/mapping.js';
import {createDemoAPI,strictJSON} from '../worker/workflow.js';
// These imports are bundled into the one-module hosted Worker at build time.
globalThis.validateMapping=validateMapping;globalThis.uncertaintyGuard=uncertaintyGuard;
const meta=JSON.parse(await readFile(new URL('../worker/model.json',import.meta.url),'utf8'));
const mapper=JSON.parse(await readFile(new URL('../worker/mapper.json',import.meta.url),'utf8'));
const replays=JSON.parse(await readFile(new URL('./guard-replay.json',import.meta.url),'utf8'));
for(const c of replays){const actual=uncertaintyGuard(c.text,c.before,meta);assert.deepEqual(actual.extraction.mentions,c.expected_mentions,c.id);assert.deepEqual(actual.checked.complaint_labels,c.expected_checked.complaint_labels,c.id);assert.equal(actual.checked.needs_review,c.expected_checked.needs_review,c.id)}
let checks=0;function check(){checks++}
const db=createLocalDB();let now=1000000,apiCalls=0,responseMode='valid';
const mention=(label,evidence,assertion='affirmed')=>({label,evidence,assertion,timing:'current_visit',subject:'patient'});
const handle=createDemoAPI(meta,mapper,createPredictor(meta),{clock:()=>now,transport:async(_url,options)=>{apiCalls++;const key=options.headers['x-goog-api-key'];assert.equal(key,'test-only');if(responseMode==='503')return new Response('{}',{status:503});if(responseMode==='429')return new Response('{}',{status:429});const text=JSON.parse(JSON.parse(options.body).contents[0].parts[0].text).nurse_note;const mentions=text.includes('may')?[mention('cc_cough','cough'),mention('cc_chestpain','chest pain')]:[mention(responseMode==='invalid'?'bad-label':'cc_chestpain','chest pain'),mention('cc_shortnessofbreath','shortness of breath','negated')];return Response.json({candidates:[{finishReason:'STOP',content:{parts:[{text:JSON.stringify({mentions,needs_review:false,review_reason:''})}]}}]})}});
const env={DB:db,GEMINI_API_KEY:'test-only'},base='https://esi.test';let cookie='';
async function call(path,body,options={}){now+=6;const headers={'Content-Type':'application/json',Origin:options.origin??base,...(cookie?{Cookie:cookie}:{})};if(options.cookie!==undefined)headers.Cookie=options.cookie;const req=new Request(base+path,{method:body===undefined?'GET':'POST',headers,body:body===undefined?undefined:typeof body==='string'?body:JSON.stringify(body)});const r=await handle(req,env);if(r.headers.has('Set-Cookie')&&options.cookie===undefined)cookie=r.headers.get('Set-Cookie').split(';')[0];return {status:r.status,body:await r.json(),headers:r.headers}}
const intake={age:46,gender:'Male',arrival:'Walk-in',complaint:'Reports chest pain; denies shortness of breath.',hr:110,rr:24,spo2:94,temp:37.2,sbp:148,dbp:92,pain:null};
let r=await call('/api/health');assert.equal(r.body.laptop_required,false);assert.equal(r.body.sign_in_required,false);check();
r=await call('/api/complaint-categories');assert.equal(r.body.categories.length,200);assert.match(r.headers.get('Set-Cookie'),/HttpOnly/);assert.ok(cookie);check();
r=await call('/api/triage/drafts',{intake,mapping_mode:'manual'},{origin:'https://other.test'});assert.equal(r.status,403);check();
for(const patch of [{age:17},{age:'46'},{hr:true},{gender:'Other'},{complaint:''},{arrival:''},{unexpected:1},{temp:'37.2'}]){r=await call('/api/triage/drafts',{intake:{...intake,...patch},mapping_mode:'manual'});assert.equal(r.status,422);check()}
r=await call('/api/triage/drafts','{"intake":{},"intake":{}}');assert.equal(r.status,400);check();
r=await call('/api/triage/drafts',' '.repeat(32769));assert.equal(r.status,413);check();
r=await call('/api/triage/drafts',{intake,mapping_mode:'manual'});assert.equal(r.status,201);assert.equal(apiCalls,0);const draft=r.body,route='/api/triage/drafts/'+draft.draft_id;check();
r=await call(route+'/score',{confirmation_token:'bad',input_digest:draft.input_digest});assert.equal(r.status,409);check();
const review={draft_revision:draft.revision,input_digest:draft.input_digest,review_confirmed:true,decisions:[],additional_confirmed_labels:['cc_chestpain']};
r=await call(route+'/confirm',{...review,additional_confirmed_labels:['bad']});assert.equal(r.status,422);check();
r=await call(route+'/confirm',review);assert.equal(r.status,200);const confirmed=r.body,scoreBody={confirmation_token:confirmed.confirmation_token,input_digest:confirmed.input_digest};check();
r=await call(route+'/score',scoreBody,{cookie:''});assert.equal(r.status,404);check();
r=await call(route+'/score',scoreBody);assert.equal(r.status,200);assert.equal(r.body.prediction.recorded_esi_prediction,2);assert.equal(r.body.esi_review_completed,false);const original=r.body.prediction;check();
r=await call(route+'/decision',{...scoreBody,decision_revision:0,action:'modify',final_esi:3});assert.equal(r.status,200);assert.equal(r.body.nurse_decision.nurse_final_esi,3);assert.deepEqual(r.body.prediction,original);check();
r=await call(route+'/decision',{...scoreBody,decision_revision:0,action:'accept'});assert.equal(r.status,409);check();
r=await call(route+'/confirm',{...review,draft_revision:confirmed.revision,review_confirmed:false});assert.equal(r.status,422);r=await call(route+'/score',scoreBody);assert.equal(r.status,409);check();
responseMode='503';r=await call('/api/triage/drafts',{intake});assert.equal(r.status,503);assert.equal(r.body.draft_id,undefined);check();
responseMode='429';r=await call('/api/triage/drafts',{intake});assert.equal(r.status,429);check();
responseMode='invalid';r=await call('/api/triage/drafts',{intake});assert.equal(r.status,502);check();
const {age,gender,arrival,complaint,...vitals}=intake;
responseMode='valid';r=await call('/api/triage/drafts',{intake:{age,gender,arrival,complaint:'Reports cough; may also have chest pain.'}});assert.equal(r.status,201);const u=r.body;assert.equal(u.mentions[1].assertion,'uncertain');assert.equal(u.intake.hr,null);check();
r=await call('/api/triage/drafts/'+u.draft_id+'/confirm',{draft_revision:u.revision,input_digest:u.input_digest,review_confirmed:true,decisions:u.mentions.map((m,i)=>({mention_id:m.mention_id,action:i?'unresolved':'confirm',...(i?{}:{label:m.label})}))});assert.equal(r.status,409);check();
const staged='/api/triage/drafts/'+u.draft_id;
const stagedReview={draft_revision:u.revision,input_digest:u.input_digest,review_confirmed:true,decisions:[{mention_id:'m0',action:'confirm',label:'cc_cough'},{mention_id:'m1',action:'exclude'}]};
r=await call(staged+'/confirm',{...stagedReview,decisions:[stagedReview.decisions[0]]});assert.equal(r.status,422);check();
r=await call(staged+'/confirm',stagedReview);assert.equal(r.status,200);assert.deepEqual(r.body.confirmed_labels,['cc_cough']);const stageToken={confirmation_token:r.body.confirmation_token,input_digest:r.body.input_digest};check();
r=await call(staged+'/score',stageToken);assert.equal(r.status,422);check();
r=await call(staged+'/score',{...stageToken,vitals:{...vitals,rr:null,spo2:null}});assert.equal(r.status,422);check();
r=await call(staged+'/score',{...stageToken,vitals:{...vitals,age:50}});assert.equal(r.status,422);check();
r=await call(staged+'/score',{...stageToken,vitals:{...vitals,spo2:null}});assert.equal(r.status,200);assert.ok(r.body.prediction.recorded_esi_prediction);const previousRevision=r.body.decision_revision;check();
r=await call(staged+'/decision',{...stageToken,decision_revision:previousRevision,action:'accept'});assert.equal(r.status,200);check();
r=await call(staged+'/score',{...stageToken,vitals:{...vitals,spo2:null,hr:130}});assert.equal(r.status,200);assert.equal(r.body.nurse_decision,null);assert.equal(r.body.esi_review_completed,false);assert.deepEqual(r.body.confirmed_labels,['cc_cough']);check();
r=await call(staged+'/decision',{...stageToken,decision_revision:previousRevision+1,action:'accept'});assert.equal(r.status,409);check();
now+=1201;r=await call(route+'/score',scoreBody);assert.equal(r.status,410);check();
db.close();
const result={workflow_checks:checks,guard_replays:replays.length,guard_mentions_labels_and_review_match:true,actual_estimator_used:true,anonymous_session_isolation:true,source_model_sha256:meta.source_sha256};
await writeFile(new URL('./cloud-check-results.json',import.meta.url),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
