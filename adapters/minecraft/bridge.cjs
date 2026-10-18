"use strict";
const readline=require('node:readline');
const {validate}=require('./policy.cjs');
if(!process.argv.includes('--authorized')) { console.error('Use --authorized only for a local server you may automate.'); process.exit(1); }
const mineflayer=require('mineflayer');
const emit=value=>process.stdout.write(JSON.stringify({time:new Date().toISOString(),...value})+'\n');
const bot=mineflayer.createBot({host:'127.0.0.1',port:25565,username:'AtlasLocal',auth:'offline'});
let paused=false,busy=false,actions=0,ready=false,movementTimer,finishMovement;
const input=readline.createInterface({input:process.stdin,crlfDelay:Infinity});
function halt(){paused=true;clearTimeout(movementTimer);bot.clearControlStates();if(finishMovement){finishMovement();finishMovement=null;}if(bot.targetDigBlock)bot.stopDigging();}
function shutdown(){halt();bot.quit();input.close();}
bot.once('spawn',()=>{ready=true;emit({event:'ready',commands:['observe','move','dig','craft','pause','resume','stop']});});
bot.on('error',e=>{emit({error:e.message});shutdown();});
bot.on('end',()=>{ready=false;input.close();});
bot.on('kicked',reason=>emit({event:'kicked',reason}));
process.on('SIGINT',shutdown);
input.on('close',()=>{halt();bot.quit();});
input.on('line',async line=>{
  if(Buffer.byteLength(line)>4096){emit({error:'Command too large'});return;}
  try {
    const c=validate(JSON.parse(line));
    if(c.action==='stop'){shutdown();return;}
    if(c.action==='pause'){halt();emit({paused:true});return;}
    if(c.action==='resume'){if(busy)throw new Error('Wait for current action to finish');paused=false;emit({paused:false});return;}
    if(!ready)throw new Error('Wait for spawn');
    if(paused)throw new Error('Bridge is paused');
    if(c.action==='observe'){emit({observation:{position:bot.entity.position,health:bot.health,food:bot.food,inventory:bot.inventory.items().map(i=>({name:i.name,count:i.count}))}});return;}
    if(busy)throw new Error('An action is already running');
    if(actions>=100)throw new Error('Session action budget reached; restart manually');
    actions++;busy=true;
    try {
      if(c.action==='move'){
        bot.setControlState(c.direction,true);
        await new Promise(resolve=>{finishMovement=resolve;movementTimer=setTimeout(()=>{bot.clearControlStates();finishMovement=null;resolve();},c.ms);});
      }else if(c.action==='dig'){
        const {Vec3}=require('vec3');const pos=new Vec3(...c.position);
        if(bot.entity.position.distanceTo(pos)>4)throw new Error('Block is outside four-block reach');
        const block=bot.blockAt(pos);
        if(!block || !['dirt','grass_block','sand','oak_log','stone'].includes(block.name))throw new Error('Block type is not on the dig allowlist');
        let timeout;
        try {await Promise.race([bot.dig(block),new Promise((_,reject)=>{timeout=setTimeout(()=>{bot.stopDigging();reject(new Error('Dig timeout'));},5000);})]);}
        finally{clearTimeout(timeout);}
      }else if(c.action==='craft'){
        const item=bot.registry.itemsByName[c.item];if(!item)throw new Error('Unknown item');
        const recipe=bot.recipesFor(item.id,null,c.count,null)[0];if(!recipe)throw new Error('No inventory-only crafting recipe available');
        let timeout;
        try {await Promise.race([bot.craft(recipe,c.count,null),new Promise((_,reject)=>{timeout=setTimeout(()=>{shutdown();reject(new Error('Craft timeout; bridge disconnected'));},5000);})]);}
        finally{clearTimeout(timeout);}
      }
      emit({action:c.action,status:paused?'interrupted':'completed',actions_used:actions});
    }finally{busy=false;bot.clearControlStates();}
  }catch(e){emit({error:e.message});}
});
