"use strict";
function validate(command) {
  const names = new Set(['observe','move','dig','craft','pause','resume','stop']);
  if (!command || !names.has(command.action)) throw new Error('Action is not allowed');
  if (command.action==='move' && (!['forward','back','left','right','jump'].includes(command.direction) || !Number.isInteger(command.ms) || command.ms<1 || command.ms>2000)) throw new Error('Movement needs a direction and 1..2000 ms');
  if (command.action==='dig' && (!Array.isArray(command.position) || command.position.length!==3 || !command.position.every(Number.isInteger))) throw new Error('Dig needs an integer [x,y,z]');
  if (command.action==='craft' && (typeof command.item!=='string' || !/^[a-z_]+$/.test(command.item) || !Number.isInteger(command.count) || command.count<1 || command.count>8)) throw new Error('Craft needs an item name and count 1..8');
  return command;
}
module.exports={validate};
