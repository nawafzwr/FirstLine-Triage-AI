import {readFile,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createPredictor} from '../worker/model-runtime.js';
const meta=JSON.parse(await readFile(new URL('../worker/model.json',import.meta.url),'utf8'));
const cases=JSON.parse(await readFile(new URL('./model-parity.json',import.meta.url),'utf8'));
const predict=createPredictor(meta);let maximum=0;
for(const c of cases){const actual=predict(c.intake,c.labels);assert.equal(actual.recorded_esi_prediction,c.prediction);Object.values(actual.probabilities).forEach((v,i)=>{maximum=Math.max(maximum,Math.abs(v-c.expected[i]));assert.ok(Math.abs(v-c.expected[i])<1e-7)})}
const result={cases:cases.length,all_predictions_match:true,max_probability_difference:maximum,source_sha256:meta.source_sha256};
await writeFile(new URL('./model-parity-result.json',import.meta.url),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
