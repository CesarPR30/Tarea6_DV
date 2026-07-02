/* =========================================================================
   NeuroScope — Hidden Activity of ANNs  (D3)
   T1 inter-epoch · T2 inter-layer · Squares-style score view · cross-linked charts
   ========================================================================= */
const C10 = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
             "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"];

const state = {
  ds:null, mode:"t1", meta:null, t1:null, t2:null,
  stageIdx:0, playing:false, timer:null,
  hidden:new Set(), selected:new Set(),
  tool:"pan", showHulls:false, showMis:false, showTrails:false,
};

const svg = d3.select("#chart");
const tip = d3.select("#tooltip");
let gZoom,gTrails,gHulls,gDots,gErr,gLasso,x,y,zoom,W,H,curK=1;

const cur = () => state.mode==="t1" ? state.t1 : state.t2;
const stages = () => cur().stages;
const stage = () => stages()[state.stageIdx];
const stageName = s => state.mode==="t1" ? `Época ${s.epoch}` : `Capa ${s.layer}`;
const nClasses = () => state.meta.classes.length;
const clsName = c => state.meta.classes[c];
const clsShort = c => { const n=clsName(c); return n.length>4? n.slice(0,3):n; };

/* ===================================================================== */
async function loadDataset(ds){
  state.ds=ds; state.selected.clear(); state.hidden.clear();
  const [meta,t1,t2]=await Promise.all([
    d3.json(`static/data/${ds}_meta.json`),d3.json(`static/data/${ds}_t1.json`),d3.json(`static/data/${ds}_t2.json`)]);
  state.meta=meta; state.t1=t1; state.t2=t2; state.stageIdx=stages().length-1;
  document.getElementById("model-info").textContent=`MLP ${meta.n_layers}×${meta.hidden} ReLU`;
  document.getElementById("acc-info").textContent=(meta.final_acc*100).toFixed(2)+"%";
  buildLegend(); setupChart(); refreshAll();
}

/* ---------- scatter scaffold ---------- */
function setupChart(){
  const wrap=document.getElementById("chart-wrap");
  W=wrap.clientWidth; H=wrap.clientHeight;
  svg.attr("viewBox",`0 0 ${W} ${H}`).selectAll("*").remove();
  const pad=26;
  x=d3.scaleLinear().domain([0,1]).range([pad,W-pad]);
  y=d3.scaleLinear().domain([0,1]).range([H-pad,pad]);
  gZoom=svg.append("g");
  gHulls=gZoom.append("g"); gTrails=gZoom.append("g"); gDots=gZoom.append("g"); gErr=gZoom.append("g");
  gLasso=svg.append("g");
  zoom=d3.zoom().scaleExtent([0.7,12]).on("zoom",e=>{
    curK=e.transform.k; gZoom.attr("transform",e.transform);
    gDots.selectAll("circle").attr("r",3.4/Math.sqrt(curK));
    gErr.selectAll("circle").attr("r",5.4/Math.sqrt(curK));
  });
  applyTool();
  svg.on("pointerdown",lassoDown).on("pointermove",lassoMove).on("pointerup",lassoUp);
}
function applyTool(){
  if(state.tool==="pan"){ svg.call(zoom); svg.classed("lasso",false); }
  else { svg.on(".zoom",null); svg.classed("lasso",true); }
  document.getElementById("lasso-hint").classList.toggle("hidden",state.tool!=="lasso");
}
function resetView(){ svg.transition().duration(450).call(zoom.transform,d3.zoomIdentity); }

/* ---------- legend ---------- */
function buildLegend(){
  const box=d3.select("#legend").html("");
  state.meta.classes.forEach((name,c)=>{
    const it=box.append("div").attr("class","legend-item").attr("data-c",c)
      .on("click",e=>{ if(e.shiftKey) toggleHide(c); else selectClass(c); })
      .on("mouseenter",()=>hoverClass(c)).on("mouseleave",()=>hoverClass(null));
    it.append("span").attr("class","legend-swatch").style("background",C10[c]);
    it.append("span").text(name);
  });
}
function toggleHide(c){
  state.hidden.has(c)?state.hidden.delete(c):state.hidden.add(c);
  d3.select(`.legend-item[data-c='${c}']`).classed("off",state.hidden.has(c));
  refreshSelectionViews(); buildStrip();
}
function hoverClass(c){ gDots.selectAll("circle").classed("dim",d=>c!==null&&d.label!==c); }

/* ===================================================================== */
function refreshAll(){
  document.getElementById("mode-desc").textContent=state.mode==="t1"
    ? "T1 — t-SNE compartido sobre la unión de todas las épocas (última capa oculta). Las clases se separan al aprender."
    : "T2 — t-SNE por capa oculta (época final), inicializadas con la capa 1. La representación se refina al fluir por la red.";
  document.getElementById("strip-kind").textContent=state.mode==="t1"?"de épocas":"de capas";
  document.getElementById("stage-label").textContent=state.mode==="t1"?"Época":"Capa";
  document.getElementById("brightness-key").textContent=state.mode==="t1"
    ? "Trayectorias: brillo ↑ = épocas tardías":"Trayectorias: brillo ↑ = capas profundas";
  document.getElementById("sq-desc").textContent=state.mode==="t1"
    ? "Eje Y = score de la red (softmax). Cada columna agrupa lo predicho como esa clase."
    : "Eje Y = acuerdo k-NN en la proyección. Cada columna agrupa lo clasificado como esa clase.";
  const sl=document.getElementById("stage-slider"); sl.max=stages().length-1; sl.value=state.stageIdx;
  const ticks=d3.select("#stage-ticks").html("");
  stages().forEach(s=>ticks.append("span").text(state.mode==="t1"?s.epoch:`L${s.layer}`));
  buildStrip(); drawStage(true); refreshSelectionViews();
}

/* ---------- helpers ---------- */
const classOn=d=>!state.hidden.has(d.label);
const selectionClass=()=>{
  if(!state.selected.size) return null;
  let c=null; for(const i of state.selected){ const l=cur().labels[i]; if(c===null)c=l; else if(c!==l)return null; }
  return c;
};
function points(){
  const s=stage();
  return s.points.map((p,i)=>({i,px:p[0],py:p[1],label:cur().labels[i],pred:s.pred[i],score:s.score?s.score[i]:1}));
}

/* ---------- scatter ---------- */
function drawStage(animate){
  const data=points(), selOn=state.selected.size>0;
  document.getElementById("big-title").textContent=
    `${state.mode==="t1"?"T1 · Evolución por épocas":"T2 · Evolución por capas"} — ${stageName(stage())}`;
  document.getElementById("nh-badge").textContent="NH "+(stage().nh*100).toFixed(1)+"%";
  const sel=gDots.selectAll("circle").data(data,d=>d.i);
  sel.exit().remove();
  const ent=sel.enter().append("circle").attr("class","dot").attr("r",3.4/Math.sqrt(curK))
    .attr("cx",d=>x(d.px)).attr("cy",d=>y(d.py))
    .on("mousemove",showTip).on("mouseleave",hideTip).on("click",(e,d)=>clickPoint(e,d));
  const merged=ent.merge(sel);
  merged.attr("fill",d=>C10[d.label])
    .style("display",d=>classOn(d)?null:"none")
    .classed("sel-dim",d=>selOn&&!state.selected.has(d.i))
    .classed("selected",d=>selOn&&state.selected.has(d.i));
  (animate?merged.transition().duration(750).ease(d3.easeCubicInOut):merged)
    .attr("cx",d=>x(d.px)).attr("cy",d=>y(d.py));
}

/* ---------- hulls (clustering por clase) ---------- */
function drawHulls(){
  gHulls.selectAll("*").remove();
  if(!state.showHulls) return;
  const s=stage();
  for(let c=0;c<nClasses();c++){
    if(state.hidden.has(c)) continue;
    const pts=[];
    cur().labels.forEach((l,i)=>{ if(l===c) pts.push([x(s.points[i][0]),y(s.points[i][1])]); });
    if(pts.length<3) continue;
    const hull=d3.polygonHull(pts); if(!hull) continue;
    gHulls.append("path").attr("class","hull")
      .attr("d","M"+hull.map(p=>p.join(",")).join("L")+"Z")
      .attr("fill",C10[c]).attr("stroke",C10[c]);
  }
}

/* ---------- error rings (mal clasificados con otro color) ---------- */
function drawErrRings(){
  gErr.selectAll("*").remove();
  if(!(state.showMis||state.showHulls)) return;
  const s=stage(), selOn=state.selected.size>0;
  s.points.forEach((p,i)=>{
    if(s.pred[i]===cur().labels[i]) return;
    if(state.hidden.has(cur().labels[i])) return;
    if(selOn&&!state.selected.has(i)) return;
    gErr.append("circle").attr("class","err-ring")
      .attr("cx",x(p[0])).attr("cy",y(p[1])).attr("r",5.4/Math.sqrt(curK));
  });
}

/* ---------- trails ---------- */
function trailColor(label,t){ const b=d3.hsl(C10[label]); b.l=0.28+0.5*t; return b.toString(); }
const mid=(a,b)=>[(a[0]+b[0])/2,(a[1]+b[1])/2];
function drawTrails(){
  gTrails.selectAll("*").remove();
  if(!state.showTrails) return;
  const sgs=stages(),n=cur().labels.length,selOn=state.selected.size>0;
  const line=d3.line().x(d=>x(d[0])).y(d=>y(d[1])).curve(d3.curveBasis);
  for(let i=0;i<n;i++){
    const label=cur().labels[i]; if(state.hidden.has(label)) continue;
    const dim=selOn&&!state.selected.has(i), path=sgs.map(s=>s.points[i]);
    for(let k=0;k<path.length-1;k++){
      gTrails.append("path").attr("class","trail")
        .attr("d",line([path[k],mid(path[k],path[k+1]),path[k+1]]))
        .attr("stroke",trailColor(label,(k+1)/(path.length-1)))
        .attr("stroke-width",dim?0.4:0.8).attr("opacity",dim?0.05:0.32).attr("stroke-linecap","round");
    }
  }
}

/* ---------- selection plumbing ---------- */
function selectClass(c){ state.selected=new Set(); cur().labels.forEach((l,i)=>{if(l===c)state.selected.add(i);}); afterSelect(); }
function selectCell(r,c){
  state.selected=new Set(); const s=stage();
  cur().labels.forEach((l,i)=>{ if(l===r&&s.pred[i]===c&&!state.hidden.has(l)) state.selected.add(i); });
  afterSelect();
}
function clickPoint(e,d){ e.stopPropagation(); state.selected=new Set([d.i]); afterSelect(); }
function clearSelection(){ state.selected.clear(); afterSelect(); }
function afterSelect(){ refreshSelectionViews(); }
function refreshSelectionViews(){
  drawStage(false); drawHulls(); drawErrRings(); drawTrails();
  drawAnalytics(); drawSquares(); updateSelInfo(); updateLegendFocus();
}
function updateSelInfo(){
  const fc=selectionClass();
  document.getElementById("sel-info").textContent =
    !state.selected.size? "— (todas)" : (fc!==null? `clase «${clsName(fc)}» (${state.selected.size})` : `${state.selected.size} obs.`);
}
function updateLegendFocus(){ const fc=selectionClass();
  d3.selectAll(".legend-item").classed("focus",function(){ return +this.dataset.c===fc; }); }

/* ---------- lasso ---------- */
let lassoPts=null,lassoPath=null;
function lassoDown(e){ if(state.tool!=="lasso")return; lassoPts=[d3.pointer(e,svg.node())]; lassoPath=gLasso.append("path").attr("class","lasso-path"); }
function lassoMove(e){ if(!lassoPts)return; lassoPts.push(d3.pointer(e,svg.node())); lassoPath.attr("d","M"+lassoPts.map(p=>p.join(",")).join("L")+"Z"); }
function lassoUp(){
  if(!lassoPts) return;
  if(lassoPts.length>3){
    const t=d3.zoomTransform(svg.node()),s=stage(),sel=new Set();
    s.points.forEach((p,i)=>{ if(state.hidden.has(cur().labels[i]))return;
      if(pointInPoly(t.applyX(x(p[0])),t.applyY(y(p[1])),lassoPts)) sel.add(i); });
    state.selected=sel; afterSelect();
  }
  if(lassoPath)lassoPath.remove(); lassoPts=null; lassoPath=null;
}
function pointInPoly(px,py,poly){ let inside=false;
  for(let i=0,j=poly.length-1;i<poly.length;j=i++){ const xi=poly[i][0],yi=poly[i][1],xj=poly[j][0],yj=poly[j][1];
    if(((yi>py)!==(yj>py))&&(px<(xj-xi)*(py-yi)/(yj-yi)+xi)) inside=!inside; } return inside; }

/* ---------- tooltip ---------- */
function spriteStyle(i){ const sp=state.meta.sprite,col=i%sp.cols,row=(i/sp.cols|0),scale=66/sp.th;
  return {backgroundImage:`url(static/data/${state.ds}_sprite.png)`,
    backgroundPosition:`-${col*sp.tw*scale}px -${row*sp.th*scale}px`,
    backgroundSize:`${sp.cols*sp.tw*scale}px ${sp.rows*sp.th*scale}px`}; }
function showTip(e,d){
  const err=d.pred!==d.label, predLbl=state.mode==="t1"?"Predicción (red)":"Clasif. k-NN proyección";
  tip.classed("hidden",false).html(`
    <div class="tt-img"></div>
    <div class="tt-row"><span>Clase real</span><b style="color:${C10[d.label]}">${clsName(d.label)}</b></div>
    <div class="tt-row"><span>${predLbl}</span><b class="${err?'tt-err':''}">${clsName(d.pred)}${err?' ✗':' ✓'}</b></div>
    <div class="tt-row"><span>Score</span><b>${(d.score*100).toFixed(0)}%</b></div>`);
  Object.assign(tip.select(".tt-img").node().style,spriteStyle(d.i));
  const r=document.getElementById("chart-wrap").getBoundingClientRect();
  let lx=e.clientX-r.left+14,ly=e.clientY-r.top+14;
  if(lx>r.width-150)lx-=180; if(ly>r.height-150)ly-=170;
  tip.style("left",lx+"px").style("top",ly+"px");
}
function hideTip(){ tip.classed("hidden",true); }

/* ---------- filmstrip ---------- */
function buildStrip(){
  const strip=d3.select("#strip").html(""),w=190,h=104,pad=6;
  const sx=d3.scaleLinear().domain([0,1]).range([pad,w-pad]),sy=d3.scaleLinear().domain([0,1]).range([h-pad,pad]);
  stages().forEach((s,k)=>{
    const cell=strip.append("div").attr("class","strip-cell"+(k===state.stageIdx?" active":""))
      .on("click",()=>{ stopPlay(); gotoStage(k,true); });
    const head=cell.append("div").attr("class","sc-head");
    head.append("span").text(state.mode==="t1"?`Época ${s.epoch}`:`Capa ${s.layer}`);
    head.append("b").text("NH "+(s.nh*100).toFixed(0)+"%");
    cell.append("svg").attr("viewBox",`0 0 ${w} ${h}`).selectAll("circle").data(s.points).join("circle")
      .attr("cx",d=>sx(d[0])).attr("cy",d=>sy(d[1])).attr("r",1.1)
      .attr("fill",(d,i)=>C10[cur().labels[i]])
      .style("display",(d,i)=>state.hidden.has(cur().labels[i])?"none":null);
  });
}

/* ===================================================================== */
/* ANALYTICS                                                             */
/* ===================================================================== */
function scopeIdx(){ const labels=cur().labels,out=[];
  for(let i=0;i<labels.length;i++){ if(state.hidden.has(labels[i]))continue;
    if(state.selected.size&&!state.selected.has(i))continue; out.push(i); } return out; }
function svgSize(id){ const n=document.getElementById(id); return [n.clientWidth,n.clientHeight]; }

function drawAnalytics(){
  const scoped=state.selected.size?`sel (${state.selected.size})`:"todas";
  document.getElementById("cm-scope").textContent=scoped;
  document.getElementById("bar-scope").textContent=scoped;
  document.getElementById("line-scope").textContent=state.mode==="t1"?"hasta época actual":"hasta capa actual";
  drawLineChart(); drawConfusion(); drawPerClass();
}

/* progressive line: only draw the trail up to the current stage */
function drawLineChart(){
  const [w,h]=svgSize("line-chart"),m={t:8,r:8,b:18,l:26},sgs=stages();
  const sv=d3.select("#line-chart").attr("viewBox",`0 0 ${w} ${h}`); sv.selectAll("*").remove();
  const xs=d3.scalePoint().domain(d3.range(sgs.length)).range([m.l,w-m.r]),ys=d3.scaleLinear().domain([0,1]).range([h-m.b,m.t]);
  [0,.25,.5,.75,1].forEach(v=>{ sv.append("line").attr("class","gridline").attr("x1",m.l).attr("x2",w-m.r).attr("y1",ys(v)).attr("y2",ys(v));
    sv.append("text").attr("class","ax").attr("x",2).attr("y",ys(v)+3).text(Math.round(v*100)); });
  sgs.forEach((s,i)=>sv.append("text").attr("class","ax").attr("x",xs(i)).attr("y",h-6).attr("text-anchor","middle")
    .attr("font-weight",i===state.stageIdx?700:400).text(state.mode==="t1"?s.epoch:`L${s.layer}`));
  const shown=sgs.slice(0,state.stageIdx+1);
  const mk=(key,color)=>{
    if(shown.length>1){ const line=d3.line().x((d,i)=>xs(i)).y(d=>ys(d[key])).curve(d3.curveMonotoneX);
      sv.append("path").attr("fill","none").attr("stroke",color).attr("stroke-width",2.2).attr("d",line(shown)); }
    sv.selectAll(null).data(shown).join("circle").attr("cx",(d,i)=>xs(i)).attr("cy",d=>ys(d[key]))
      .attr("r",(d,i)=>i===state.stageIdx?3.6:2.4).attr("fill",color);
  };
  mk("nh","#5b3df5"); mk("acc","#16a34a");
  sv.append("line").attr("x1",xs(state.stageIdx)).attr("x2",xs(state.stageIdx)).attr("y1",m.t).attr("y2",h-m.b)
    .attr("stroke","#b9c0d8").attr("stroke-dasharray","3 3");
  sv.append("circle").attr("cx",m.l+4).attr("cy",m.t+4).attr("r",3).attr("fill","#5b3df5");
  sv.append("text").attr("class","ax").attr("x",m.l+10).attr("y",m.t+7).text("NH");
  sv.append("circle").attr("cx",m.l+32).attr("cy",m.t+4).attr("r",3).attr("fill","#16a34a");
  sv.append("text").attr("class","ax").attr("x",m.l+38).attr("y",m.t+7).text("Acc");
  sv.append("rect").attr("x",m.l).attr("y",m.t).attr("width",w-m.l-m.r).attr("height",h-m.t-m.b).attr("fill","transparent")
    .style("cursor","pointer").on("click",e=>{ const mx=d3.pointer(e)[0]; let best=0,bd=1e9;
      sgs.forEach((s,i)=>{const dd=Math.abs(xs(i)-mx); if(dd<bd){bd=dd;best=i;}}); stopPlay(); gotoStage(best,true); });
}

function confusionMatrix(){
  const k=nClasses(),M=Array.from({length:k},()=>new Array(k).fill(0)),labels=cur().labels,preds=stage().pred;
  scopeIdx().forEach(i=>{ M[labels[i]][preds[i]]++; }); return M;
}
let cmActive=null;
function drawConfusion(){
  const [w,h]=svgSize("cm-chart"),k=nClasses(),m={t:14,r:6,b:16,l:18};
  const sv=d3.select("#cm-chart").attr("viewBox",`0 0 ${w} ${h}`); sv.selectAll("*").remove();
  const cell=Math.min((w-m.l-m.r)/k,(h-m.t-m.b)/k),ox=m.l,oy=m.t,M=confusionMatrix(),maxv=d3.max(M.flat())||1;
  const col=d3.scaleSequential(d3.interpolatePuBu).domain([0,maxv]);
  for(let r=0;r<k;r++)for(let c=0;c<k;c++){
    sv.append("rect").attr("class","cm-cell"+(r===c?" cm-diag":"")+(cmActive&&cmActive[0]===r&&cmActive[1]===c?" cm-active":""))
      .attr("x",ox+c*cell).attr("y",oy+r*cell).attr("width",cell).attr("height",cell)
      .attr("fill",M[r][c]?col(M[r][c]):"#f6f8fd")
      .on("click",()=>{ if(M[r][c]===0)return; cmActive=[r,c]; selectCell(r,c); })
      .append("title").text(`real ${clsName(r)} → pred ${clsName(c)}: ${M[r][c]}`);
  }
  for(let i=0;i<k;i++){
    sv.append("text").attr("class","ax").attr("x",ox+i*cell+cell/2).attr("y",oy-4).attr("text-anchor","middle")
      .attr("fill",C10[i]).attr("font-weight",600).text(clsShort(i));
    sv.append("text").attr("class","ax").attr("x",ox-3).attr("y",oy+i*cell+cell/2+3).attr("text-anchor","end")
      .attr("fill",C10[i]).attr("font-weight",600).text(clsShort(i));
  }
  sv.append("text").attr("class","ax").attr("x",ox+k*cell/2).attr("y",h-3).attr("text-anchor","middle").text("predicho →");
}

function drawPerClass(){
  const [w,h]=svgSize("bar-chart"),k=nClasses(),m={t:6,r:26,b:14,l:42};
  const sv=d3.select("#bar-chart").attr("viewBox",`0 0 ${w} ${h}`); sv.selectAll("*").remove();
  const M=confusionMatrix(),recall=M.map((row,r)=>{const s=d3.sum(row);return s?row[r]/s:null;});
  const ys=d3.scaleBand().domain(d3.range(k)).range([m.t,h-m.b]).padding(.18),xs=d3.scaleLinear().domain([0,1]).range([m.l,w-m.r]);
  [0,.5,1].forEach(v=>sv.append("line").attr("class","gridline").attr("x1",xs(v)).attr("x2",xs(v)).attr("y1",m.t).attr("y2",h-m.b));
  recall.forEach((v,c)=>{
    sv.append("text").attr("class","ax").attr("x",m.l-4).attr("y",ys(c)+ys.bandwidth()/2+3).attr("text-anchor","end")
      .attr("fill",C10[c]).text(clsShort(c));
    if(v===null)return;
    sv.append("rect").attr("x",xs(0)).attr("y",ys(c)).attr("height",ys.bandwidth()).attr("width",Math.max(0,xs(v)-xs(0)))
      .attr("fill",C10[c]).attr("opacity",.85).style("cursor","pointer").on("click",()=>selectClass(c))
      .append("title").text(`${clsName(c)}: recall ${(v*100).toFixed(1)}%`);
    sv.append("text").attr("class","ax").attr("x",xs(v)+3).attr("y",ys(c)+ys.bandwidth()/2+3).text((v*100).toFixed(0));
  });
}

/* ===================================================================== */
/* SQUARES-style per-class score distribution                            */
/* ===================================================================== */
function drawSquares(){
  const node=document.getElementById("squares"); const w=node.clientWidth||900,h=280;
  const sv=d3.select("#squares").attr("viewBox",`0 0 ${w} ${h}`); sv.selectAll("*").remove();
  const k=nClasses(),m={t:16,r:14,b:26,l:34};
  const s=stage(),labels=cur().labels,NB=20,fc=selectionClass(),selOn=state.selected.size>0;
  const cx=d3.scaleBand().domain(d3.range(k)).range([m.l,w-m.r]).paddingInner(.35);
  const cy=d3.scaleLinear().domain([0,1]).range([h-m.b,m.t]);
  // y axis
  [0,.2,.4,.6,.8,1].forEach(v=>{ sv.append("line").attr("class","gridline").attr("x1",m.l).attr("x2",w-m.r).attr("y1",cy(v)).attr("y2",cy(v));
    sv.append("text").attr("class","ax").attr("x",4).attr("y",cy(v)+3).text(v.toFixed(1)); });
  sv.append("text").attr("class","ax").attr("transform",`translate(11,${(h-m.b+m.t)/2}) rotate(-90)`).attr("text-anchor","middle").text("score →");

  // bin instances by predicted column
  const cols=d3.range(k).map(()=>Array.from({length:NB},()=>({correct:0,mis:[]})));
  for(let i=0;i<labels.length;i++){
    const tl=labels[i]; if(state.hidden.has(tl))continue;
    const pc=s.pred[i], sc=Math.max(0,Math.min(.999,s.score?s.score[i]:1)), b=Math.floor(sc*NB);
    if(tl===pc) cols[pc][b].correct++; else cols[pc][b].mis.push({i,tl});
  }
  const maxStack=d3.max(cols.flat().map(c=>c.correct+c.mis.length))||1;
  const bw=cx.bandwidth(), unit=Math.max(.7,(bw-2)/maxStack), binH=(h-m.b-m.t)/NB;
  const op=i=>!selOn?1:(state.selected.has(i)?1:.12);

  for(let c=0;c<k;c++){
    const x0=cx(c);
    sv.append("line").attr("class","sq-axis").attr("x1",x0).attr("x2",x0).attr("y1",m.t).attr("y2",h-m.b).attr("stroke","#e4e8f3");
    sv.append("text").attr("class","sq-col-label").attr("x",x0+bw/2).attr("y",h-10).attr("text-anchor","middle")
      .attr("fill",C10[c]).text(clsShort(c)).on("click",()=>selectClass(c)).append("title").text(`Predicho como ${clsName(c)} — clic para resaltar la clase real`);
    for(let b=0;b<NB;b++){
      const cell=cols[c][b]; if(!cell.correct&&!cell.mis.length)continue;
      const yTop=cy((b+1)/NB)+1, hh=Math.max(1,binH-2);
      if(cell.correct>0){
        sv.append("rect").attr("class","sq-bar").attr("x",x0+1).attr("y",yTop).attr("width",cell.correct*unit).attr("height",hh)
          .attr("fill",C10[c]).attr("opacity",selOn?(fc===c?1:.5):.92).on("click",()=>selectClass(c))
          .append("title").text(`${clsName(c)} correctos · score ${(b/NB).toFixed(2)}–${((b+1)/NB).toFixed(2)}: ${cell.correct}`);
      }
      cell.mis.forEach((mm,j)=>{
        sv.append("rect").attr("class","sq-stripe").attr("x",x0+1+cell.correct*unit+j*unit).attr("y",yTop)
          .attr("width",Math.max(unit-.6,.8)).attr("height",hh)
          .attr("fill",C10[mm.tl]).attr("fill-opacity",.32).attr("stroke",C10[mm.tl]).attr("opacity",op(mm.i))
          .style("cursor","pointer").on("click",()=>selectClass(mm.tl))
          .append("title").text(`Real ${clsName(mm.tl)} → predicho ${clsName(c)} · score ${(b/NB).toFixed(2)}`);
      });
    }
  }
  // connector lines for the focused true class (where its instances landed)
  if(fc!==null){
    const pts=[];
    for(let i=0;i<labels.length;i++){ if(labels[i]!==fc)continue; const pc=s.pred[i],sc=s.score?s.score[i]:1;
      pts.push([cx(pc)+bw/2, cy(sc)]); }
    pts.sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
    if(pts.length>1){
      const line=d3.line().curve(d3.curveCatmullRom.alpha(.5));
      sv.append("path").attr("class","sq-link").attr("d",line(pts)).attr("stroke",C10[fc]);
    }
    sv.selectAll(null).data(pts).join("circle").attr("cx",d=>d[0]).attr("cy",d=>d[1]).attr("r",2).attr("fill",C10[fc]).attr("opacity",.7);
  }
}

/* ---------- navigation ---------- */
function gotoStage(k,animate){
  state.stageIdx=k; document.getElementById("stage-slider").value=k;
  d3.selectAll(".strip-cell").classed("active",(d,i)=>i===k);
  cmActive=null; drawStage(animate); drawHulls(); drawErrRings(); drawTrails(); drawAnalytics(); drawSquares();
}
function playEvolution(){
  if(state.playing){ stopPlay(); return; }
  state.playing=true; const b=document.getElementById("play-btn"); b.classList.add("playing"); b.textContent="⏸ Detener";
  gotoStage(0,false); let k=0;
  state.timer=setInterval(()=>{ k++; if(k>=stages().length){ stopPlay(); return; } gotoStage(k,true); },1300);
}
function stopPlay(){ state.playing=false; clearInterval(state.timer);
  const b=document.getElementById("play-btn"); b.classList.remove("playing"); b.textContent="▶ Animar evolución"; }

/* ---------- wiring ---------- */
function setMode(m){ if(state.mode===m)return; stopPlay(); state.mode=m; cmActive=null;
  document.getElementById("mode-t1").classList.toggle("active",m==="t1");
  document.getElementById("mode-t2").classList.toggle("active",m==="t2");
  state.stageIdx=stages().length-1; refreshAll(); }
function setTool(t){ state.tool=t;
  document.getElementById("tool-pan").classList.toggle("active",t==="pan");
  document.getElementById("tool-lasso").classList.toggle("active",t==="lasso"); applyTool(); }

document.getElementById("mode-t1").onclick=()=>setMode("t1");
document.getElementById("mode-t2").onclick=()=>setMode("t2");
document.getElementById("tool-pan").onclick=()=>setTool("pan");
document.getElementById("tool-lasso").onclick=()=>setTool("lasso");
document.getElementById("clear-sel").onclick=clearSelection;
document.getElementById("dataset").onchange=e=>loadDataset(e.target.value);
document.getElementById("stage-slider").oninput=e=>{ stopPlay(); gotoStage(+e.target.value,true); };
document.getElementById("play-btn").onclick=playEvolution;
document.getElementById("reset-view").onclick=resetView;
document.getElementById("legend-all").onclick=()=>{ state.hidden.clear(); d3.selectAll(".legend-item").classed("off",false);
  buildStrip(); refreshSelectionViews(); };
document.getElementById("show-hulls").onchange=e=>{ state.showHulls=e.target.checked; drawHulls(); drawErrRings(); };
document.getElementById("show-mis").onchange=e=>{ state.showMis=e.target.checked; drawErrRings(); };
document.getElementById("show-trails").onchange=e=>{ state.showTrails=e.target.checked; drawTrails(); };
window.addEventListener("resize",()=>{ if(state.meta){ setupChart(); drawStage(false); refreshSelectionViews(); }});

loadDataset(window.DATASETS[0]);
