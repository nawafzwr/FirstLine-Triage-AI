// Inference-only reader for the unchanged fitted sklearn estimator.
export function createPredictor(meta) {
  let view;
  function nodes(){if(!view){const binary=atob(meta.binary_base64),bytes=new Uint8Array(binary.length);for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);view=new DataView(bytes.buffer)}return view}
  return function predict(intake,labels){
    const selected=new Set(labels);
    if(!selected.size||[...selected].some(x=>!meta.labels.includes(x)))throw new Error('Explicit known complaint categories are required.');
    const raw=[intake.age,intake.hr,intake.sbp,intake.dbp,intake.rr,intake.spo2,intake.temp,intake.arrival==='Ambulance'?1:intake.arrival==='Walk-in'?0:null];
    const missing=raw.map(x=>x==null);
    if(raw.slice(1,7).filter(x=>x==null).length>1)throw new Error('At most one vital may be missing.');
    const features=raw.map((x,i)=>x==null?meta.medians[i]:x);
    features.push(...meta.missing_indicators.map(i=>Number(missing[i])));
    const gender=['Female','Male'].includes(intake.gender)?intake.gender:'Unknown';
    features.push(...meta.gender_categories.map(x=>Number(gender===x)));
    features.push(...meta.labels.map(x=>Number(selected.has(x))),0);
    const families=new Set([...selected].map(x=>meta.families[x]));
    const route=families.size===1?[...families][0]:'multiple_families';
    const classifier=meta.models[route]||meta.models.shared;
    const scores=classifier.baseline.slice(),data=nodes();
    for(const [start,count,k] of classifier.trees){
      let pos=0;
      for(let depth=0;depth<count;depth++){
        const offset=(start+pos)*21,feature=data.getUint16(offset,true);
        if(feature===65535){scores[k]+=data.getFloat64(offset+13,true);break}
        const value=features[feature],left=data.getUint8(offset+10),right=data.getUint8(offset+11);
        pos=Number.isNaN(value)?(data.getUint8(offset+12)?left:right):value<=data.getFloat64(offset+2,true)?left:right;
      }
    }
    const max=Math.max(...scores),exp=scores.map(x=>Math.exp(x-max)),total=exp.reduce((a,b)=>a+b,0),p=[0,0,0,0,0];
    classifier.classes.forEach((label,i)=>p[label-1]=Math.fround(exp[i]/total));
    const chosen=p.indexOf(Math.max(...p))+1;
    const coverage=[];
    meta.numeric_features.forEach((key,i)=>{const b=meta.observed_ranges[key],v=raw[i];if(v!=null&&b.min!=null&&(v<b.min||v>b.max))coverage.push(`${key} is outside the observed training range.`)});
    if(gender==='Unknown')coverage.push('Unknown/Other gender has no observed training examples.');
    return {recorded_esi_prediction:chosen,argmax_prediction:chosen,probabilities:Object.fromEntries(p.map((x,i)=>[String(i+1),x])),decision_rule:'argmax',undertriage_penalty:null,probabilities_calibrated:false,unused_prototype_fields:['pain','complaint narrative'],coverage_notes:coverage,scope:'Research model for adults; category labels must be explicitly confirmed.'};
  }
}
