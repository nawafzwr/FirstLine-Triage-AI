export function validateMapping(text,extraction,meta){
  const exact=(value,keys)=>value&&typeof value==='object'&&!Array.isArray(value)&&Object.keys(value).sort().join('|')===keys.slice().sort().join('|');
  if(!exact(extraction,['mentions','needs_review','review_reason'])||typeof extraction.needs_review!=='boolean'||typeof extraction.review_reason!=='string'||!Array.isArray(extraction.mentions)||extraction.mentions.length>40)throw new Error('Invalid extraction structure.');
  if(extraction.needs_review!==Boolean(extraction.review_reason.trim()))throw new Error('Inconsistent review reason.');
  const labels=new Set(),evidence=new Map(),reasons=[];
  for(const m of extraction.mentions){
    if(!exact(m,['label','evidence','assertion','timing','subject'])||(m.label!==null&&!meta.labels.includes(m.label))||typeof m.evidence!=='string'||!m.evidence.trim()||!text.includes(m.evidence)||!['affirmed','negated','uncertain'].includes(m.assertion)||!['current_visit','historical_only','uncertain'].includes(m.timing)||!['patient','other','uncertain'].includes(m.subject))throw new Error('Invalid extraction mention or evidence.');
    if(m.assertion==='negated'||m.timing==='historical_only'||m.subject==='other')continue;
    if([m.assertion,m.timing,m.subject].includes('uncertain')){reasons.push('A potentially relevant mention has uncertain context.');continue}
    if(m.label===null){reasons.push('A current visit reason has no resolved model category.');continue}
    if(evidence.has(m.evidence)&&evidence.get(m.evidence)!==m.label)reasons.push('The same evidence was assigned multiple categories.');
    evidence.set(m.evidence,m.label);labels.add(m.label);
  }
  const ordered=meta.labels.filter(x=>labels.has(x));
  if(!ordered.length)reasons.push('No affirmed patient complaint for this visit was mapped.');
  if(extraction.needs_review)reasons.push(extraction.review_reason.trim());
  return {complaint_labels:ordered,needs_review:reasons.length>0,review_reasons:[...new Set(reasons)],ready_for_model:reasons.length===0};
}

export function uncertaintyGuard(text,original,meta){
  validateMapping(text,original,meta);
  const extraction=structuredClone(original),changes=[],reasons=[];
  const trigger=/\b(?:(?:may|might|could)(?:\s+also)?\s+(?:have|be|has|having)|possible|possibly|maybe|perhaps|(?:unsure|uncertain)\s+(?:if|whether|about having))\b/i;
  const eligible=m=>m.assertion!=='negated'&&m.timing!=='historical_only'&&m.subject!=='other';
  if(trigger.test(text)){
    const tokens=[...text.matchAll(/[\p{L}\p{N}_]+(?:['’][\p{L}]+)?|[^\s\p{L}\p{N}_]/gu)];
    const starts=new Set(tokens.map(x=>x.index)),ends=new Set(tokens.map(x=>x.index+x[0].length));
    for(const [i,m] of extraction.mentions.entries()){
      if(!eligible(m)||m.assertion!=='affirmed')continue;
      const start=text.indexOf(m.evidence),end=start+m.evidence.length;
      if(text.indexOf(m.evidence,start+1)!==-1){changes.push({mention_index:i,action:'review_ambiguous_evidence'});reasons.push('Repeated evidence needs occurrence clarification before confirming uncertainty.');continue}
      if(!starts.has(start)||!ends.has(end)){changes.push({mention_index:i,action:'review_unaligned_evidence'});reasons.push('Evidence cannot be aligned exactly for the uncertainty check.');continue}
      const sentenceStart=Math.max(...['.','!','?',';','\n'].map(c=>text.lastIndexOf(c,start-1)))+1;
      const prefix=text.slice(sentenceStart,start);
      const cues=[...prefix.matchAll(/\b(?:(?:may|might|could)(?:\s+also)?\s+(?:have|be|has|having)|possible|possibly|maybe|perhaps|(?:unsure|uncertain)\s+(?:if|whether|about having))\b/gi)];
      const matching=cues.filter(c=>!(/\b(?:but|however|yet|except|reports|endorses|complains\s+of)\b/i.test(prefix.slice(c.index+c[0].length))));
      if(matching.length){m.assertion='uncertain';changes.push({mention_index:i,action:'affirmed_to_uncertain',evidence:m.evidence,source_modifiers:matching.map(c=>({text:c[0],category:'POSSIBLE_EXISTENCE'}))});reasons.push('Explicit symptom uncertainty requires nurse confirmation.')}
    }
  }
  if(extraction.mentions.some(m=>eligible(m)&&[m.assertion,m.timing,m.subject].includes('uncertain'))&&!extraction.needs_review){changes.push({action:'require_review_for_structured_uncertainty'});reasons.push('An unresolved patient complaint requires nurse confirmation.')}
  if(reasons.length){extraction.needs_review=true;extraction.review_reason=[...new Set([extraction.review_reason.trim(),...reasons].filter(Boolean))].join(' ')}
  return {extraction,checked:validateMapping(text,extraction,meta),guard_changes:changes};
}
