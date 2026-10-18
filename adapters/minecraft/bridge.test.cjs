const test=require('node:test');
const assert=require('node:assert/strict');
const {validate}=require('./policy.cjs');
test('movement and crafting are bounded',()=>{
 assert.equal(validate({action:'move',direction:'forward',ms:2000}).ms,2000);
 for(const c of [{action:'shell'},{action:'move',direction:'forward',ms:2001},{action:'move',direction:'attack',ms:20},{action:'craft',item:'oak_planks',count:100},{action:'dig',position:[1,2,3.5]}])assert.throws(()=>validate(c));
});
test('emergency controls are always valid',()=>{for(const action of ['pause','resume','stop'])assert.equal(validate({action}).action,action);});
