import { useState, useEffect, useRef, useCallback } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

// ══════════════════════════════════════════════════════════════════════════════
// Brand tokens
// ══════════════════════════════════════════════════════════════════════════════
const B = {
  blue:   '#416BA9',
  blueDark: '#2f4f82',
  blueLight: '#d6e3f3',
  blueMid:  '#5b87c5',
  gray:   '#BBBBBB',
  grayLight: '#F5F5F5',
  grayMid: '#E0E0E0',
  charcoal: '#222222',
  teal:   '#9FCFCA',
  tealDark: '#5aada6',
  purple: '#B14FC5',
  pink:   '#FEB0C0',
  white:  '#ffffff',
  font:   "'Gotham', 'Segoe UI', Arial, sans-serif",
};

// ══════════════════════════════════════════════════════════════════════════════
// Constants
// ══════════════════════════════════════════════════════════════════════════════
const MODES = { FUNC: "func", TSP: "tsp" };
const DOMAIN = [-5, 5];
const CW = 460, CH = 360;
const BITS = 16;
const MAX_INT = (1 << BITS) - 1;

const rnd = (a, b) => Math.random() * (b - a) + a;
const rndInt = (a, b) => Math.floor(Math.random() * (b - a + 1)) + a;

// ══════════════════════════════════════════════════════════════════════════════
// Fitness landscape
// ══════════════════════════════════════════════════════════════════════════════
const landscape = (x, y) =>
  2.6 * Math.exp(-((x-1)**2+(y-1)**2)/1.8) + 2.1 * Math.exp(-((x+2.5)**2+(y-2)**2)/1.1) +
  1.8 * Math.exp(-((x-3.2)**2+(y+2.5)**2)/0.9) + 1.6 * Math.exp(-((x+1)**2+(y+3.5)**2)/1.3) +
  0.5 * Math.sin(2*x) * Math.cos(2*y);

const buildHeatmap = (w, h) => {
  const data = new Uint8ClampedArray(w*h*4), dr = DOMAIN[1]-DOMAIN[0];
  const vals = new Float32Array(w*h); let lo=Infinity, hi=-Infinity;
  for (let py=0; py<h; py++) for (let px=0; px<w; px++) {
    const v = landscape(DOMAIN[0]+(px/w)*dr, DOMAIN[1]-(py/h)*dr);
    vals[py*w+px]=v; if(v<lo) lo=v; if(v>hi) hi=v;
  }
  // Brand palette: light blue-gray → mid blue → brand blue → teal accent at peaks
  for (let i=0; i<w*h; i++) {
    const t = (vals[i]-lo)/(hi-lo); let r,g,b;
    if (t<0.33) { const s=t/0.33; r=Math.floor(214+s*(159-214)); g=Math.floor(227+s*(191-227)); b=Math.floor(245+s*(228-245)); }
    else if (t<0.66) { const s=(t-0.33)/0.33; r=Math.floor(159+s*(65-159)); g=Math.floor(191+s*(107-191)); b=Math.floor(228+s*(169-228)); }
    else { const s=(t-0.66)/0.34; r=Math.floor(65+s*(159-65)); g=Math.floor(107+s*(207-107)); b=Math.floor(169+s*(202-169)); }
    data[i*4]=r; data[i*4+1]=g; data[i*4+2]=b; data[i*4+3]=255;
  }
  return data;
};

// ══════════════════════════════════════════════════════════════════════════════
// Binary encoding
// ══════════════════════════════════════════════════════════════════════════════
const bitsToInt = bits => bits.reduce((s,b,i)=>s+(b<<(BITS-1-i)),0);
const bitsToVal = bits => DOMAIN[0]+(bitsToInt(bits)/MAX_INT)*(DOMAIN[1]-DOMAIN[0]);
const randomBits = () => Array.from({length:BITS},()=>rndInt(0,1));

// ══════════════════════════════════════════════════════════════════════════════
// Func GA
// ══════════════════════════════════════════════════════════════════════════════
const mkFuncInd = () => ({xBits:randomBits(),yBits:randomBits(),fitness:0,xVal:0,yVal:0});
const evalFunc = pop => pop.map(ind => {
  const xVal=bitsToVal(ind.xBits), yVal=bitsToVal(ind.yBits);
  return {...ind,xVal,yVal,fitness:landscape(xVal,yVal)};
});
const tournament = (pop,k=3) => {
  let best=null;
  for(let i=0;i<k;i++){const c=pop[rndInt(0,pop.length-1)];if(!best||c.fitness>best.fitness)best=c;}
  return best;
};
const crossoverFunc = (a,b) => {
  const as=[...a.xBits,...a.yBits], bs=[...b.xBits,...b.yBits];
  const pt=rndInt(1,BITS*2-1);
  const cs=as.map((bit,i)=>i<pt?bit:bs[i]);
  return {xBits:cs.slice(0,BITS),yBits:cs.slice(BITS),fitness:0,xVal:0,yVal:0,_pt:pt};
};
const mutateFunc = (ind,rate) => {
  const mb=[];
  const nx=ind.xBits.map((b,i)=>{if(Math.random()<rate){mb.push(i);return 1-b;}return b;});
  const ny=ind.yBits.map((b,i)=>{if(Math.random()<rate){mb.push(BITS+i);return 1-b;}return b;});
  return {...ind,xBits:nx,yBits:ny,_mb:mb};
};
const stepFunc = (pop,params) => {
  const sorted=[...pop].sort((a,b)=>b.fitness-a.fitness);
  const ec=Math.max(1,Math.floor(params.elitism*pop.length));
  const next=sorted.slice(0,ec).map(i=>({...i}));
  let xEv=null, mEv=null;
  while(next.length<pop.length){
    const pA=tournament(pop), pB=tournament(pop);
    let child=crossoverFunc(pA,pB);
    if(!xEv) xEv={
      pA:{xBits:[...pA.xBits],yBits:[...pA.yBits],fitness:pA.fitness,xVal:pA.xVal,yVal:pA.yVal},
      pB:{xBits:[...pB.xBits],yBits:[...pB.yBits],fitness:pB.fitness,xVal:pB.xVal,yVal:pB.yVal},
      pt:child._pt, child:{xBits:[...child.xBits],yBits:[...child.yBits]}
    };
    const bef={xBits:[...child.xBits],yBits:[...child.yBits]};
    child=mutateFunc(child,params.mutationRate);
    if(!mEv&&child._mb?.length>0) mEv={before:bef,after:{xBits:[...child.xBits],yBits:[...child.yBits]},mb:child._mb};
    next.push(child);
  }
  return {pop:evalFunc(next),xEv,mEv};
};

// ══════════════════════════════════════════════════════════════════════════════
// TSP GA
// ══════════════════════════════════════════════════════════════════════════════
const genCities = n => Array.from({length:n},()=>({x:rnd(40,CW-40),y:rnd(40,CH-40)}));
const tspDist = (cities,route) => {
  let d=0;
  for(let i=0;i<route.length;i++){const a=cities[route[i]],b=cities[route[(i+1)%route.length]];d+=Math.hypot(a.x-b.x,a.y-b.y);}
  return d;
};
const mkTSPInd = n => {
  const r=Array.from({length:n},(_,i)=>i);
  for(let i=r.length-1;i>0;i--){const j=rndInt(0,i);[r[i],r[j]]=[r[j],r[i]];}
  return {route:r,fitness:0,dist:0};
};
const evalTSP = (pop,cities) => pop.map(ind=>({...ind,fitness:10000/tspDist(cities,ind.route),dist:tspDist(cities,ind.route)}));
const crossoverTSP = (a,b) => {
  const n=a.route.length, s=rndInt(0,n-2), e=rndInt(s+1,n-1);
  const child=new Array(n).fill(-1);
  for(let i=s;i<=e;i++) child[i]=a.route[i];
  const used=new Set(child.filter(v=>v>=0)); let bi=0;
  for(let i=0;i<n;i++){if(child[i]===-1){while(used.has(b.route[bi]))bi++;child[i]=b.route[bi++];used.add(child[i]);}}
  return {route:child,fitness:0,dist:0,_s:s,_e:e};
};
const mutateTSP = (ind,rate) => {
  const route=[...ind.route],sw=[];
  for(let i=0;i<route.length;i++){
    if(Math.random()<rate){const j=rndInt(0,route.length-1);if(i!==j){sw.push(i,j);[route[i],route[j]]=[route[j],route[i]];}}
  }
  return {route,fitness:0,dist:0,_sw:sw};
};
const stepTSP = (pop,cities,params) => {
  const sorted=[...pop].sort((a,b)=>b.fitness-a.fitness);
  const ec=Math.max(1,Math.floor(params.elitism*pop.length));
  const next=sorted.slice(0,ec).map(i=>({...i,route:[...i.route]}));
  let xEv=null, mEv=null;
  while(next.length<pop.length){
    const pA=tournament(pop), pB=tournament(pop);
    let child=crossoverTSP(pA,pB);
    if(!xEv) xEv={
      pA:{route:[...pA.route],fitness:pA.fitness,dist:pA.dist},
      pB:{route:[...pB.route],fitness:pB.fitness,dist:pB.dist},
      s:child._s,e:child._e,child:{route:[...child.route]}
    };
    const bef={route:[...child.route]};
    child=mutateTSP(child,params.mutationRate);
    if(!mEv&&child._sw?.length>0) mEv={before:bef,after:{route:[...child.route]},sw:[...new Set(child._sw)]};
    next.push(child);
  }
  return {pop:evalTSP(next,cities),xEv,mEv};
};

// ══════════════════════════════════════════════════════════════════════════════
// Canvas drawing
// ══════════════════════════════════════════════════════════════════════════════
const drawFunc = (ctx,pop,heatmap) => {
  if(heatmap) ctx.drawImage(heatmap,0,0,CW,CH);
  const dr=DOMAIN[1]-DOMAIN[0];
  const best=[...pop].sort((a,b)=>b.fitness-a.fitness)[0];
  pop.forEach(ind=>{
    const px=((ind.xVal-DOMAIN[0])/dr)*CW, py=((DOMAIN[1]-ind.yVal)/dr)*CH;
    ctx.beginPath(); ctx.arc(px,py,ind===best?8:3.5,0,Math.PI*2);
    ctx.fillStyle = ind===best ? B.purple : 'rgba(34,34,34,0.55)';
    ctx.fill();
    if(ind===best){ctx.strokeStyle=B.white;ctx.lineWidth=2;ctx.stroke();}
  });
};
const drawTSP = (ctx,pop,cities) => {
  ctx.fillStyle='#f0f4fa'; ctx.fillRect(0,0,CW,CH);
  // Subtle grid
  ctx.strokeStyle='rgba(65,107,169,0.07)'; ctx.lineWidth=1;
  for(let x=0;x<CW;x+=40){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,CH);ctx.stroke();}
  for(let y=0;y<CH;y+=40){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(CW,y);ctx.stroke();}
  if(!cities.length||!pop.length) return;
  const sorted=[...pop].sort((a,b)=>b.fitness-a.fitness);
  // Faint population routes
  sorted.slice(1,10).forEach(ind=>{
    ctx.beginPath();
    ind.route.forEach((ci,i)=>{const c=cities[ci];i===0?ctx.moveTo(c.x,c.y):ctx.lineTo(c.x,c.y);});
    ctx.closePath(); ctx.strokeStyle='rgba(65,107,169,0.10)'; ctx.lineWidth=1; ctx.stroke();
  });
  // Best route — teal accent
  ctx.beginPath();
  sorted[0].route.forEach((ci,i)=>{const c=cities[ci];i===0?ctx.moveTo(c.x,c.y):ctx.lineTo(c.x,c.y);});
  ctx.closePath(); ctx.strokeStyle=B.tealDark; ctx.lineWidth=2.5; ctx.stroke();
  // Cities
  cities.forEach((c,i)=>{
    ctx.beginPath(); ctx.arc(c.x,c.y,6,0,Math.PI*2);
    ctx.fillStyle=B.blue; ctx.fill();
    ctx.strokeStyle=B.white; ctx.lineWidth=1.5; ctx.stroke();
    ctx.fillStyle=B.charcoal;
    ctx.font=`bold 8px ${B.font}`; ctx.fillText(i,c.x+9,c.y+4);
  });
};

// ══════════════════════════════════════════════════════════════════════════════
// Chromosome: binary
// ══════════════════════════════════════════════════════════════════════════════
const BitChromosome = ({xBits,yBits,pt,mutBits,label,fitness,xVal,yVal}) => {
  const renderBit = (bit,gi) => {
    const fromA = pt!=null ? gi<pt : null;
    const isMut = mutBits?.includes(gi);
    let bg, border, color;
    if (isMut) { bg='#fde8ec'; border=B.pink; color='#8b0000'; }
    else if (fromA===true) { bg=B.blueLight; border=B.blue; color=B.blueDark; }
    else if (fromA===false) { bg='#e8f5f4'; border=B.teal; color=B.tealDark; }
    else { bg=B.grayLight; border=B.grayMid; color: bit?B.charcoal:B.gray; }
    return (
      <div key={gi} title={`bit ${gi} = ${bit}`} style={{
        width:11,height:20,flexShrink:0, background:bg,
        border:`1px solid ${border}`,
        display:'flex',alignItems:'center',justifyContent:'center',
        fontSize:8,color:bit?(isMut?'#8b0000':fromA===true?B.blueDark:fromA===false?B.tealDark:B.charcoal):B.gray,
        borderRadius:2, fontFamily:'monospace', fontWeight:bit?700:400,
      }}>{bit}</div>
    );
  };
  const cutMark = (
    <div style={{display:'flex',flexDirection:'column',alignItems:'center',padding:'0 2px',flexShrink:0}}>
      <span style={{fontSize:9,color:B.purple,lineHeight:'10px'}}>✂</span>
      <div style={{width:2,height:16,background:B.purple,borderRadius:1,opacity:0.6}}/>
    </div>
  );
  const xEl=[],yEl=[],gapEl=[];
  (xBits||[]).forEach((bit,i)=>{
    if(pt!=null&&i===pt) xEl.push(<span key={`cx${i}`}>{cutMark}</span>);
    xEl.push(renderBit(bit,i));
  });
  if(pt!=null&&pt===BITS) gapEl.push(<span key="cm">{cutMark}</span>);
  (yBits||[]).forEach((bit,i)=>{
    const gi=BITS+i;
    if(pt!=null&&gi===pt) yEl.push(<span key={`cy${i}`}>{cutMark}</span>);
    yEl.push(renderBit(bit,gi));
  });
  return (
    <div style={{marginBottom:12}}>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:4}}>
        <span style={{fontSize:10,color:B.blue,textTransform:'uppercase',letterSpacing:'0.08em',fontWeight:700,fontFamily:B.font}}>{label}</span>
        <div style={{display:'flex',gap:12}}>
          {xVal!=null&&<span style={{fontSize:9,color:B.gray,fontFamily:'monospace'}}>({xVal.toFixed(3)}, {yVal.toFixed(3)})</span>}
          {fitness!=null&&<span style={{fontSize:9,color:B.purple,fontFamily:'monospace',fontWeight:700}}>f={fitness.toFixed(3)}</span>}
        </div>
      </div>
      <div style={{display:'flex',alignItems:'center',overflowX:'auto',paddingBottom:2}}>
        <span style={{fontSize:8,color:B.gray,marginRight:3,flexShrink:0,fontFamily:'monospace'}}>X─</span>
        {xEl}
        <span style={{width:6,flexShrink:0}}/>{gapEl}
        <span style={{fontSize:8,color:B.gray,marginRight:3,flexShrink:0,fontFamily:'monospace'}}>Y─</span>
        {yEl}
      </div>
    </div>
  );
};

// ══════════════════════════════════════════════════════════════════════════════
// Chromosome: TSP permutation
// ══════════════════════════════════════════════════════════════════════════════
const RouteChromosome = ({route,label,fitness,dist,s,e,role,swapped}) => (
  <div style={{marginBottom:12}}>
    <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:4}}>
      <span style={{fontSize:10,color:B.blue,textTransform:'uppercase',letterSpacing:'0.08em',fontWeight:700,fontFamily:B.font}}>{label}</span>
      <div style={{display:'flex',gap:12}}>
        {dist!=null&&<span style={{fontSize:9,color:B.tealDark,fontFamily:'monospace',fontWeight:700}}>dist={dist.toFixed(0)}</span>}
        {fitness!=null&&<span style={{fontSize:9,color:B.purple,fontFamily:'monospace',fontWeight:700}}>f={fitness.toFixed(3)}</span>}
      </div>
    </div>
    <div style={{display:'flex',gap:3,flexWrap:'wrap'}}>
      {(route||[]).map((city,i)=>{
        const inSeg=s!=null&&i>=s&&i<=e, isSw=swapped?.includes(i);
        let bg=B.grayLight, border=B.grayMid, color=B.gray;
        if(isSw){bg='#fde8ec';border=B.pink;color='#8b0000';}
        else if(role==='a'&&inSeg){bg=B.blueLight;border=B.blue;color=B.blueDark;}
        else if(role==='b'&&!inSeg){bg='#e8f5f4';border=B.teal;color=B.tealDark;}
        else if(role==='child'){
          if(inSeg){bg=B.blueLight;border=B.blue;color=B.blueDark;}
          else{bg='#e8f5f4';border=B.teal;color=B.tealDark;}
        }
        return (
          <div key={i} title={`pos ${i}: city ${city}`} style={{
            width:26,height:26,background:bg,border:`1px solid ${border}`,
            display:'flex',alignItems:'center',justifyContent:'center',
            fontSize:9,color,borderRadius:4,fontFamily:'monospace',fontWeight:700,
          }}>{city}</div>
        );
      })}
    </div>
    {role&&s!=null&&(
      <div style={{fontSize:8,color:B.gray,marginTop:4,fontFamily:B.font}}>
        {role==='a'&&`Positions [${s}–${e}] → carried into child as fixed segment`}
        {role==='b'&&`Remaining cities → their relative order fills the gaps`}
        {role==='child'&&`[${s}–${e}] from Parent A · remaining from Parent B (order preserved)`}
      </div>
    )}
  </div>
);

// ══════════════════════════════════════════════════════════════════════════════
// Inspector panel
// ══════════════════════════════════════════════════════════════════════════════
const Inspector = ({mode,xEv,mEv}) => {
  const [tab,setTab]=useState('x');
  const tabBtn = (id,label,active) => (
    <button onClick={()=>setTab(id)} style={{
      padding:'7px 16px', border:`1px solid ${tab===id?B.blue:B.grayMid}`,
      cursor:'pointer', fontSize:11, fontFamily:B.font, fontWeight:tab===id?700:400,
      background:tab===id?B.blue:B.white, color:tab===id?B.white:B.gray,
      borderRadius:5, transition:'all 0.15s',
    }}>{label}</button>
  );
  const divider = <div style={{borderTop:`1px solid ${B.grayMid}`,margin:'12px 0'}}/>;
  const empty = msg => (
    <div style={{color:B.gray,fontSize:11,fontFamily:B.font,padding:'24px 0',textAlign:'center'}}>
      {msg}
    </div>
  );
  return (
    <div style={{background:B.white,border:`1px solid ${B.grayMid}`,borderRadius:10,padding:'16px 20px',marginTop:16,boxShadow:'0 2px 8px rgba(65,107,169,0.08)'}}>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:14}}>
        <div style={{display:'flex',alignItems:'center',gap:8}}>
          <div style={{width:3,height:18,background:B.blue,borderRadius:2}}/>
          <span style={{fontSize:12,fontWeight:700,color:B.charcoal,fontFamily:B.font,letterSpacing:'0.04em',textTransform:'uppercase'}}>
            Chromosome Inspector
          </span>
        </div>
        <div style={{display:'flex',gap:6}}>
          {tabBtn('x','✂  Crossover')}
          {tabBtn('m','⚡  Mutation')}
        </div>
      </div>

      {tab==='x'&&(xEv?(
        <div>
          <div style={{fontSize:10,color:B.gray,marginBottom:12,fontFamily:B.font,background:B.grayLight,padding:'6px 10px',borderRadius:5,borderLeft:`3px solid ${B.blue}`}}>
            {mode===MODES.FUNC
              ?`Single-point crossover · cut at bit ${xEv.pt} of ${BITS*2} · bits 0–${xEv.pt-1} from Parent A, bits ${xEv.pt}–${BITS*2-1} from Parent B`
              :`OX1 order crossover · segment [${xEv.s}–${xEv.e}] inherited from Parent A, remaining order from Parent B`}
            {xEv.gen!=null&&<span style={{color:B.blue,fontWeight:700}}> · Gen {xEv.gen}</span>}
          </div>
          {mode===MODES.FUNC?(
            <>
              <BitChromosome label="Parent A" xBits={xEv.pA.xBits} yBits={xEv.pA.yBits} fitness={xEv.pA.fitness} xVal={xEv.pA.xVal} yVal={xEv.pA.yVal}/>
              <BitChromosome label="Parent B" xBits={xEv.pB.xBits} yBits={xEv.pB.yBits} fitness={xEv.pB.fitness} xVal={xEv.pB.xVal} yVal={xEv.pB.yVal}/>
              {divider}
              <BitChromosome label="Child (after crossover)" xBits={xEv.child.xBits} yBits={xEv.child.yBits} pt={xEv.pt}/>
              <div style={{display:'flex',gap:16,flexWrap:'wrap',fontSize:9,color:B.gray,fontFamily:B.font,marginTop:8}}>
                <span><span style={{color:B.blue,fontWeight:700}}>■</span> bits from Parent A</span>
                <span><span style={{color:B.tealDark,fontWeight:700}}>■</span> bits from Parent B</span>
                <span><span style={{color:B.purple,fontWeight:700}}>✂</span> crossover point</span>
                <span>X gene = bits 0–{BITS-1} · Y gene = bits {BITS}–{BITS*2-1}</span>
              </div>
            </>
          ):(
            <>
              <RouteChromosome label="Parent A" route={xEv.pA.route} fitness={xEv.pA.fitness} dist={xEv.pA.dist} s={xEv.s} e={xEv.e} role="a"/>
              <RouteChromosome label="Parent B" route={xEv.pB.route} fitness={xEv.pB.fitness} dist={xEv.pB.dist} s={xEv.s} e={xEv.e} role="b"/>
              {divider}
              <RouteChromosome label="Child (after crossover)" route={xEv.child.route} s={xEv.s} e={xEv.e} role="child"/>
              <div style={{display:'flex',gap:16,flexWrap:'wrap',fontSize:9,color:B.gray,fontFamily:B.font,marginTop:8}}>
                <span><span style={{color:B.blue,fontWeight:700}}>■</span> segment from Parent A</span>
                <span><span style={{color:B.tealDark,fontWeight:700}}>■</span> remaining (Parent B order)</span>
              </div>
            </>
          )}
        </div>
      ):empty('Press RUN or STEP to capture a crossover event'))}

      {tab==='m'&&(mEv?(
        <div>
          <div style={{fontSize:10,color:B.gray,marginBottom:12,fontFamily:B.font,background:B.grayLight,padding:'6px 10px',borderRadius:5,borderLeft:`3px solid ${B.pink}`}}>
            {mode===MODES.FUNC
              ?`Bit-flip mutation · ${mEv.mb.length} bit${mEv.mb.length!==1?'s':''} flipped independently`
              :`Swap mutation · ${Math.floor(mEv.sw.length/2)} position pair${mEv.sw.length>2?'s':''} exchanged`}
            {mEv.gen!=null&&<span style={{color:B.blue,fontWeight:700}}> · Gen {mEv.gen}</span>}
          </div>
          {mode===MODES.FUNC?(
            <>
              <BitChromosome label="Before mutation" xBits={mEv.before.xBits} yBits={mEv.before.yBits}/>
              <BitChromosome label="After mutation" xBits={mEv.after.xBits} yBits={mEv.after.yBits} mutBits={mEv.mb}/>
              <div style={{display:'flex',gap:16,flexWrap:'wrap',fontSize:9,color:B.gray,fontFamily:B.font,marginTop:8}}>
                <span><span style={{color:'#c0002a',fontWeight:700}}>■</span> flipped bit (0→1 or 1→0)</span>
                <span>Each of the {BITS*2} bits flips independently at the set mutation rate</span>
              </div>
            </>
          ):(
            <>
              <RouteChromosome label="Before mutation" route={mEv.before.route}/>
              <RouteChromosome label="After mutation" route={mEv.after.route} swapped={mEv.sw}/>
              <div style={{display:'flex',gap:16,flexWrap:'wrap',fontSize:9,color:B.gray,fontFamily:B.font,marginTop:8}}>
                <span><span style={{color:'#c0002a',fontWeight:700}}>■</span> swapped positions</span>
                <span>Each position independently draws a random swap partner</span>
              </div>
            </>
          )}
        </div>
      ):empty('No mutation captured yet — try increasing mutation rate above 5%'))}
    </div>
  );
};

// ══════════════════════════════════════════════════════════════════════════════
// Shared UI
// ══════════════════════════════════════════════════════════════════════════════
const ChartTooltip = ({active,payload,label}) => {
  if(!active||!payload?.length) return null;
  return (
    <div style={{background:B.white,border:`1px solid ${B.grayMid}`,borderRadius:6,padding:'8px 12px',fontSize:12,fontFamily:B.font,boxShadow:'0 2px 8px rgba(0,0,0,0.1)'}}>
      <p style={{color:B.gray,margin:'0 0 4px',fontWeight:700}}>Gen {label}</p>
      {payload.map(p=><p key={p.name} style={{color:p.color,margin:'2px 0',fontFamily:'monospace'}}>{p.name}: {p.value.toFixed(4)}</p>)}
    </div>
  );
};

const Slider = ({label,value,min,max,step,format,onChange}) => (
  <div style={{marginBottom:14}}>
    <div style={{display:'flex',justifyContent:'space-between',marginBottom:5}}>
      <span style={{fontSize:11,color:B.charcoal,letterSpacing:'0.04em',textTransform:'uppercase',fontFamily:B.font,fontWeight:700}}>{label}</span>
      <span style={{fontSize:11,color:B.blue,fontFamily:'monospace',fontWeight:700}}>{format?format(value):value}</span>
    </div>
    <input type="range" min={min} max={max} step={step} value={value} onChange={e=>onChange(+e.target.value)}
      style={{width:'100%',accentColor:B.blue,height:4,cursor:'pointer'}}/>
  </div>
);

const Stat = ({label,value,accent}) => (
  <div style={{background:B.white,border:`1px solid ${B.grayMid}`,borderRadius:8,padding:'10px 14px',boxShadow:'0 1px 3px rgba(65,107,169,0.06)'}}>
    <div style={{fontSize:10,color:B.gray,textTransform:'uppercase',letterSpacing:'0.08em',marginBottom:4,fontFamily:B.font,fontWeight:700}}>{label}</div>
    <div style={{fontSize:22,fontFamily:'monospace',fontWeight:700,color:accent||B.blue}}>{value}</div>
  </div>
);

// ══════════════════════════════════════════════════════════════════════════════
// Main
// ══════════════════════════════════════════════════════════════════════════════
export default function GeneticViz() {
  const [mode,setMode]=useState(MODES.FUNC);
  const [running,setRunning]=useState(false);
  const [generation,setGeneration]=useState(0);
  const [popSnapshot,setPopSnapshot]=useState([]);
  const [history,setHistory]=useState([]);
  const [params,setParams]=useState({populationSize:60,mutationRate:0.05,elitism:0.10});
  const [speed,setSpeed]=useState(80);
  const [lastXEv,setLastXEv]=useState(null);
  const [lastMEv,setLastMEv]=useState(null);

  const canvasRef=useRef(null), heatmapRef=useRef(null);
  const popRef=useRef([]), citiesRef=useRef([]);
  const genRef=useRef(0), histRef=useRef([]);
  const runRef=useRef(false), intervalRef=useRef(null);
  const modeRef=useRef(mode), paramsRef=useRef(params), speedRef=useRef(speed);
  modeRef.current=mode; paramsRef.current=params; speedRef.current=speed;

  useEffect(()=>{
    const off=document.createElement('canvas'); off.width=CW; off.height=CH;
    const ctx=off.getContext('2d'); const img=ctx.createImageData(CW,CH);
    img.data.set(buildHeatmap(CW,CH)); ctx.putImageData(img,0,0);
    heatmapRef.current=off;
  },[]);

  const redraw=useCallback(()=>{
    const canvas=canvasRef.current; if(!canvas) return;
    const ctx=canvas.getContext('2d'); ctx.clearRect(0,0,CW,CH);
    if(modeRef.current===MODES.FUNC) drawFunc(ctx,popRef.current,heatmapRef.current);
    else drawTSP(ctx,popRef.current,citiesRef.current);
  },[]);

  const init=useCallback((newMode)=>{
    clearInterval(intervalRef.current); runRef.current=false; setRunning(false);
    const m=newMode??modeRef.current, size=paramsRef.current.populationSize;
    let pop;
    if(m===MODES.FUNC){pop=evalFunc(Array.from({length:size},mkFuncInd));citiesRef.current=[];}
    else{const cs=genCities(16);citiesRef.current=cs;pop=evalTSP(Array.from({length:size},()=>mkTSPInd(cs.length)),cs);}
    popRef.current=pop; genRef.current=0; histRef.current=[];
    setGeneration(0); setPopSnapshot(pop); setHistory([]); setLastXEv(null); setLastMEv(null);
    setTimeout(redraw,0);
  },[redraw]);

  useEffect(()=>{init(mode);},[mode]); // eslint-disable-line

  const doStep=useCallback(()=>{
    const p=paramsRef.current,m=modeRef.current;
    const result=m===MODES.FUNC?stepFunc(popRef.current,p):stepTSP(popRef.current,citiesRef.current,p);
    popRef.current=result.pop; genRef.current+=1; const gen=genRef.current;
    if(result.xEv) setLastXEv({...result.xEv,gen});
    if(result.mEv) setLastMEv({...result.mEv,gen});
    const sorted=[...result.pop].sort((a,b)=>b.fitness-a.fitness);
    const bestF=sorted[0].fitness, avgF=result.pop.reduce((s,i)=>s+i.fitness,0)/result.pop.length;
    histRef.current=[...histRef.current.slice(-299),{gen,best:+bestF.toFixed(5),avg:+avgF.toFixed(5)}];
    setGeneration(gen); setPopSnapshot([...result.pop]); setHistory([...histRef.current]);
    redraw();
  },[redraw]);

  const toggleRun=()=>{
    if(runRef.current){clearInterval(intervalRef.current);runRef.current=false;setRunning(false);}
    else{runRef.current=true;setRunning(true);intervalRef.current=setInterval(doStep,speedRef.current);}
  };

  useEffect(()=>{
    if(runRef.current){clearInterval(intervalRef.current);intervalRef.current=setInterval(doStep,speed);}
  },[speed,doStep]);

  const best=popSnapshot.length?[...popSnapshot].sort((a,b)=>b.fitness-a.fitness)[0]:null;

  return (
    <div style={{fontFamily:B.font,background:B.grayLight,color:B.charcoal,minHeight:'100vh',padding:'24px 16px'}}>

      {/* Header */}
      <div style={{background:B.blue,borderRadius:12,padding:'18px 28px',marginBottom:22,display:'flex',alignItems:'center',justifyContent:'space-between',boxShadow:'0 4px 16px rgba(65,107,169,0.25)'}}>
        <div>
          <div style={{fontSize:10,letterSpacing:'0.25em',color:'rgba(255,255,255,0.6)',textTransform:'uppercase',marginBottom:4,fontWeight:700}}>Evolutionary Computation Lab</div>
          <h1 style={{margin:0,fontSize:22,fontWeight:700,color:B.white,letterSpacing:'-0.01em'}}>
            Genetic Algorithm Visualiser
          </h1>
        </div>
        <div style={{textAlign:'right'}}>
          <div style={{fontSize:10,color:'rgba(255,255,255,0.5)',letterSpacing:'0.1em',textTransform:'uppercase'}}>Mode</div>
          <div style={{display:'flex',gap:4,marginTop:6}}>
            {[{id:MODES.FUNC,label:'Function'},{id:MODES.TSP,label:'Travelling Salesman'}].map(m=>(
              <button key={m.id} onClick={()=>setMode(m.id)} style={{
                padding:'6px 12px',borderRadius:5,border:`1px solid ${mode===m.id?B.white:'rgba(255,255,255,0.3)'}`,
                cursor:'pointer',fontSize:11,fontFamily:B.font,fontWeight:700,
                background:mode===m.id?B.white:'transparent',
                color:mode===m.id?B.blue:'rgba(255,255,255,0.75)',transition:'all 0.15s',
              }}>{m.label}</button>
            ))}
          </div>
        </div>
      </div>

      {/* Main layout */}
      <div style={{display:'flex',gap:16,justifyContent:'center',flexWrap:'wrap',alignItems:'flex-start'}}>

        {/* Canvas */}
        <div>
          <div style={{background:B.white,borderRadius:10,padding:4,border:`1px solid ${B.grayMid}`,boxShadow:'0 2px 12px rgba(65,107,169,0.10)'}}>
            <canvas ref={canvasRef} width={CW} height={CH} style={{display:'block',borderRadius:8}}/>
          </div>
          <div style={{fontSize:10,color:B.gray,textAlign:'center',marginTop:6,fontFamily:B.font}}>
            {mode===MODES.FUNC
              ?'Heatmap: light blue-gray = low fitness → dark blue = high fitness → teal = peak · purple dot = best individual'
              :'Teal = best route · faint blue = population · blue dots = cities'}
          </div>
        </div>

        {/* Controls */}
        <div style={{width:248,display:'flex',flexDirection:'column',gap:12}}>
          {/* Stats */}
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:8}}>
            <Stat label="Generation" value={generation}/>
            <Stat label="Pop Size" value={params.populationSize}/>
            <Stat label="Best Fitness" value={best?best.fitness.toFixed(3):'—'} accent={B.purple}/>
            {mode===MODES.TSP
              ?<Stat label="Best Distance" value={best?.dist?best.dist.toFixed(1):'—'} accent={B.tealDark}/>
              :<Stat label="Avg Fitness" value={history.length?history[history.length-1].avg.toFixed(3):'—'} accent={B.gray}/>
            }
          </div>

          {mode===MODES.FUNC&&best&&(
            <div style={{background:B.blueLight,border:`1px solid ${B.blue}`,borderRadius:8,padding:'10px 14px',fontSize:11}}>
              <div style={{color:B.blue,marginBottom:4,textTransform:'uppercase',letterSpacing:'0.08em',fontSize:10,fontWeight:700}}>Best Individual</div>
              <div style={{color:B.blueDark,fontFamily:'monospace'}}>x = {best.xVal?.toFixed(4)}</div>
              <div style={{color:B.blueDark,fontFamily:'monospace'}}>y = {best.yVal?.toFixed(4)}</div>
            </div>
          )}

          {/* Parameters */}
          <div style={{background:B.white,border:`1px solid ${B.grayMid}`,borderRadius:10,padding:'16px',boxShadow:'0 1px 4px rgba(65,107,169,0.06)'}}>
            <div style={{display:'flex',alignItems:'center',gap:6,marginBottom:14}}>
              <div style={{width:3,height:14,background:B.blue,borderRadius:2}}/>
              <span style={{fontSize:11,color:B.charcoal,textTransform:'uppercase',letterSpacing:'0.08em',fontWeight:700,fontFamily:B.font}}>Parameters</span>
            </div>
            <Slider label="Population" value={params.populationSize} min={10} max={150} step={5} onChange={v=>setParams(p=>({...p,populationSize:v}))}/>
            <Slider label="Mutation Rate" value={params.mutationRate} min={0.001} max={0.5} step={0.001} format={v=>(v*100).toFixed(1)+'%'} onChange={v=>setParams(p=>({...p,mutationRate:v}))}/>
            <Slider label="Elitism" value={params.elitism} min={0} max={0.5} step={0.01} format={v=>(v*100).toFixed(0)+'%'} onChange={v=>setParams(p=>({...p,elitism:v}))}/>
            <Slider label="Speed" value={speed} min={20} max={500} step={10} format={v=>v<=20?'Max':`${v}ms`} onChange={v=>setSpeed(v)}/>
          </div>

          {/* Buttons */}
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:8}}>
            <button onClick={toggleRun} style={{
              gridColumn:'1/-1',padding:'11px 0',border:'none',borderRadius:8,cursor:'pointer',
              fontWeight:700,fontSize:13,fontFamily:B.font,letterSpacing:'0.04em',
              background:running?'#c0002a':B.blue,color:B.white,
              boxShadow:`0 2px 8px ${running?'rgba(192,0,42,0.3)':'rgba(65,107,169,0.3)'}`,transition:'all 0.15s',
            }}>{running?'⏸  Pause':'▶  Run'}</button>
            <button onClick={()=>{if(!running)doStep();}} style={{
              padding:'9px 0',border:`1px solid ${B.grayMid}`,borderRadius:8,cursor:'pointer',
              fontWeight:700,fontSize:12,fontFamily:B.font,background:B.white,color:B.blue,transition:'all 0.15s',
            }}>⏭ Step</button>
            <button onClick={()=>init()} style={{
              padding:'9px 0',border:`1px solid ${B.grayMid}`,borderRadius:8,cursor:'pointer',
              fontWeight:700,fontSize:12,fontFamily:B.font,background:B.white,color:B.gray,transition:'all 0.15s',
            }}>↺ Reset</button>
          </div>

          {/* Tip */}
          <div style={{background:B.blueLight,border:`1px solid ${B.blue}`,borderRadius:8,padding:'10px 14px',fontSize:11,color:B.blueDark,lineHeight:1.6,fontFamily:B.font}}>
            {mode===MODES.FUNC
              ?<><strong>Tip:</strong> Use Step mode to study individual crossover events in the inspector below. Try slowing speed to 300ms+.</>
              :<><strong>Tip:</strong> OX1 must re-order cities to keep the route valid. Notice how the blue segment transfers intact, then Parent B fills the gaps.</>
            }
          </div>
        </div>
      </div>

      {/* Inspector */}
      <div style={{maxWidth:760,margin:'0 auto'}}>
        <Inspector mode={mode} xEv={lastXEv} mEv={lastMEv}/>
      </div>

      {/* Chart */}
      {history.length>1&&(
        <div style={{maxWidth:760,margin:'14px auto 0',background:B.white,border:`1px solid ${B.grayMid}`,borderRadius:10,padding:'16px 20px',boxShadow:'0 2px 8px rgba(65,107,169,0.07)'}}>
          <div style={{display:'flex',alignItems:'center',gap:6,marginBottom:14}}>
            <div style={{width:3,height:14,background:B.blue,borderRadius:2}}/>
            <span style={{fontSize:11,color:B.charcoal,textTransform:'uppercase',letterSpacing:'0.08em',fontWeight:700,fontFamily:B.font}}>Fitness Over Generations</span>
          </div>
          <ResponsiveContainer width="100%" height={155}>
            <LineChart data={history} margin={{top:4,right:16,bottom:4,left:0}}>
              <CartesianGrid strokeDasharray="3 4" stroke={B.grayMid} vertical={false}/>
              <XAxis dataKey="gen" stroke={B.grayMid} tick={{fill:B.gray,fontSize:10,fontFamily:'monospace'}}/>
              <YAxis stroke={B.grayMid} tick={{fill:B.gray,fontSize:10,fontFamily:'monospace'}} width={55}/>
              <Tooltip content={<ChartTooltip/>}/>
              <Legend wrapperStyle={{fontSize:11,fontFamily:B.font,color:B.gray}}/>
              <Line type="monotone" dataKey="best" stroke={B.purple} dot={false} strokeWidth={2.5} name="Best"/>
              <Line type="monotone" dataKey="avg" stroke={B.teal} dot={false} strokeWidth={1.5} strokeDasharray="5 3" name="Average"/>
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Footer */}
      <div style={{textAlign:'center',marginTop:20,fontSize:10,color:B.gray,letterSpacing:'0.12em',fontFamily:B.font}}>
        FUNC: 32-BIT BINARY CHROMOSOME · SINGLE-POINT CROSSOVER · BIT-FLIP MUTATION &nbsp;·&nbsp; TSP: PERMUTATION · OX1 CROSSOVER · SWAP MUTATION
      </div>
    </div>
  );
}
