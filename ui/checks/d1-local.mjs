import {DatabaseSync} from 'node:sqlite';
import {readFileSync} from 'node:fs';
export function createLocalDB(path=':memory:'){
 const db=new DatabaseSync(path);
 if(!db.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='esi_sessions'").get())db.exec(readFileSync(new URL('../drizzle/0000_anonymous_demo.sql',import.meta.url),'utf8').replaceAll('--> statement-breakpoint',''));
 function prepare(sql,params=[]){return {bind(...args){return prepare(sql,args)},async first(){return db.prepare(sql).get(...params)??null},async run(){return {meta:db.prepare(sql).run(...params)}},async all(){return {results:db.prepare(sql).all(...params)}}}}
 return {prepare,async batch(items){db.exec('BEGIN');try{const results=[];for(const item of items)results.push(await item.run());db.exec('COMMIT');return results}catch(e){db.exec('ROLLBACK');throw e}},close(){db.close()}};
}
