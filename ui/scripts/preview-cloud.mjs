import {createServer} from 'node:http';
import worker from '../dist/server/index.js';
import {createLocalDB} from '../checks/d1-local.mjs';
const DB=createLocalDB(),port=Number(process.env.ESI_PREVIEW_PORT||8778);
const server=createServer(async(req,res)=>{try{const chunks=[];for await(const c of req)chunks.push(c);const body=Buffer.concat(chunks);const response=await worker.fetch(new Request(`http://127.0.0.1:${port}${req.url}`,{method:req.method,headers:req.headers,...(body.length?{body}:{})}),{DB});res.writeHead(response.status,Object.fromEntries(response.headers));res.end(Buffer.from(await response.arrayBuffer()))}catch{res.writeHead(500);res.end('Preview failed.')}});
server.listen(port,'127.0.0.1',()=>console.log(`ESI hosted-runtime preview: http://127.0.0.1:${port}`));
