const page = /* PAGE_HTML */ "";
/* MODEL_RUNTIME */
/* MAPPING_RUNTIME */
/* WORKFLOW_RUNTIME */
const meta = /* MODEL_DATA */ null;
const mapper = /* MAPPER_CONFIG */ null;
const apiHandler=createDemoAPI(meta,mapper,createPredictor(meta));
export default {async fetch(request,env){
 const url=new URL(request.url);
 if(url.pathname==='/'&&request.method==='GET')return new Response(page,{headers:{'Content-Type':'text/html; charset=utf-8','Cache-Control':'no-store','X-Content-Type-Options':'nosniff','Referrer-Policy':'same-origin'}});
 if(url.pathname==='/favicon.ico')return new Response(null,{status:204});
 if(url.pathname.startsWith('/api/'))return apiHandler(request,env);
 return Response.json({error:{code:'not_found',message:'Unknown route.'}},{status:404});
}};
