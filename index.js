const API = 'https://uk.wikipedia.org/w/api.php';

async function fetchSuggestions(q){
  if(!q.trim()) return [];
  const url = `${API}?action=opensearch&search=${encodeURIComponent(q)}&limit=6&namespace=0&format=json&origin=*`;
  const res = await fetch(url);
  const data = await res.json();
  return data[1] || [];
}

/* ЄДИНЕ МІСЦЕ ДЛЯ ЗАМІНИ ДЖЕРЕЛА НА GOOGLE/BING/SERPAPI */
async function fetchResults(q){
  const url = `${API}?action=query&list=search&srsearch=${encodeURIComponent(q)}&srlimit=8&format=json&origin=*&srprop=snippet|wordcount|timestamp`;
  const res = await fetch(url);
  const data = await res.json();
  return (data.query && data.query.search) || [];
}

/* ========= НЕЙРОМЕРЕЖА: 4 входи -> 4 приховані -> 1 вихід =========
   Входи на кожен результат:
   1. позиція у вихідній видачі (інвертована, 1..0)
   2. довжина заголовка, нормалізована (коротший — зазвичай точніший)
   3. кількість слів запиту, що трапилися в заголовку+сніпеті (частка)
   4. «свіжість»/розмір статті (нормалізований wordcount)
   Навчається на кліках: обраний результат -> ціль 1, решта показаних -> ціль 0.15 */
const STORAGE_KEY = 'neuro-web-search-weights-v1';
function randW(n){ return Array.from({length:n},()=> (Math.random()*2-1)*0.5); }
function loadWeights(){
  try{ const raw=localStorage.getItem(STORAGE_KEY); if(raw) return JSON.parse(raw); }catch(e){}
  return { W1:[randW(4),randW(4),randW(4),randW(4)], b1:randW(4), W2:randW(4), b2:(Math.random()*2-1)*0.3, trained:0 };
}
function saveWeights(w){ try{ localStorage.setItem(STORAGE_KEY, JSON.stringify(w)); }catch(e){} }
let NET = loadWeights();
function sigmoid(x){ return 1/(1+Math.exp(-x)); }
function forward(x, net){
  const h=[];
  for(let i=0;i<4;i++){
    let sum=net.b1[i];
    for(let j=0;j<4;j++) sum+=net.W1[i][j]*x[j];
    h.push(sigmoid(sum));
  }
  let outSum=net.b2;
  for(let i=0;i<4;i++) outSum+=net.W2[i]*h[i];
  return { out:sigmoid(outSum), h };
}
function trainStep(x, target, net, lr=0.35){
  const {out,h}=forward(x,net);
  const err=target-out, dOut=err*out*(1-out);
  for(let i=0;i<4;i++){
    net.W2[i]+=lr*dOut*h[i];
    const dH=dOut*net.W2[i]*h[i]*(1-h[i]);
    for(let j=0;j<4;j++) net.W1[i][j]+=lr*dH*x[j];
    net.b1[i]+=lr*dH;
  }
  net.b2+=lr*dOut;
}

function stripTags(html){ return html.replace(/<[^>]+>/g,''); }

function featuresFor(item, index, total, queryTokens){
  const posScore = 1 - (index/total);
  const titleLen = Math.min(1, item.title.length/60);
  const text = (item.title+' '+stripTags(item.snippet||'')).toLowerCase();
  const hitCount = queryTokens.filter(t=>text.includes(t)).length;
  const coverage = queryTokens.length ? hitCount/queryTokens.length : 0;
  const sizeNorm = Math.min(1,(item.wordcount||0)/3000);
  return [posScore, 1-titleLen, coverage, sizeNorm];
}

/* ========= UI ========= */
const input=document.getElementById('q');
const resultsEl=document.getElementById('results');
const countEl=document.getElementById('count');
const suggEl=document.getElementById('suggestions');
const dotEl=document.getElementById('dot');
const netstateEl=document.getElementById('netstate');
const resetBtn=document.getElementById('resetBtn');
let activeIndex=-1, currentResults=[], requestSeq=0;

function renderSuggestions(list){
  if(!list.length){ suggEl.innerHTML=''; return; }
  suggEl.innerHTML=list.map(s=>`<span class="chip">${s}</span>`).join('');
  [...suggEl.children].forEach((chip,i)=>{
    chip.addEventListener('click', ()=>{
      input.value=list[i];
      runSearch(list[i]);
    });
  });
}

function render(items){
  currentResults=items; activeIndex=-1;
  resultsEl.innerHTML='';
  if(!items.length){
    resultsEl.innerHTML = '<div class="empty">Нічого не знайдено (або немає з’єднання з інтернетом).</div>';
    countEl.textContent='0 результатів';
    return;
  }
  countEl.textContent = items.length + ' результатів';
  items.forEach((r,i)=>{
    const div=document.createElement('div');
    div.className='result';
    div.innerHTML=`
      <div class="top-row"><span class="title">${r.title}</span><span class="score">${(r.out*100).toFixed(0)}%</span></div>
      <div class="url">uk.wikipedia.org/wiki/${encodeURIComponent(r.title.replace(/ /g,'_'))}</div>
      <div class="snippet">${r.snippet}</div>`;
    div.addEventListener('click', ()=>choose(i));
    div.addEventListener('mouseenter', ()=>setActive(i));
    resultsEl.appendChild(div);
  });
}
function setActive(i){
  activeIndex=i;
  [...resultsEl.children].forEach((el,idx)=>el.classList.toggle('active', idx===activeIndex));
}
function choose(i){
  const chosen=currentResults[i];
  currentResults.forEach((r,idx)=> trainStep(r.feats, idx===i?1:0.15, NET));
  NET.trained=(NET.trained||0)+1;
  saveWeights(NET);
  const url = `https://uk.wikipedia.org/wiki/${encodeURIComponent(chosen.title.replace(/ /g,'_'))}`;
  window.open(url, '_blank');
}

async function runSearch(q){
  const seq=++requestSeq;
  suggEl.innerHTML='';
  if(!q.trim()){ render([]); resultsEl.innerHTML=''; countEl.textContent=''; return; }
  dotEl.classList.add('busy');
  countEl.textContent='шукаю…';
  try{
    const raw = await fetchResults(q);
    if(seq!==requestSeq) return; // прийшла застаріла відповідь — ігноруємо
    const queryTokens = q.toLowerCase().split(/\s+/).filter(Boolean);
    const scored = raw.map((item,i)=>{
      const feats = featuresFor(item, i, raw.length, queryTokens);
      const {out} = forward(feats, NET);
      return { title:item.title, snippet:item.snippet, feats, out };
    });
    scored.sort((a,b)=>b.out-a.out);
    render(scored);
  }catch(e){
    if(seq!==requestSeq) return;
    resultsEl.innerHTML = '<div class="empty">Не вдалося отримати дані. Перевірте інтернет-з’єднання.</div>';
    countEl.textContent='';
  }finally{
    if(seq===requestSeq) dotEl.classList.remove('busy');
  }
}

let debounceSearch=null, debounceSugg=null;
input.addEventListener('input', ()=>{
  const value=input.value;
  clearTimeout(debounceSugg);
  debounceSugg=setTimeout(async ()=>{
    try{ renderSuggestions(await fetchSuggestions(value)); }catch(e){}
  }, 150);
  clearTimeout(debounceSearch);
  debounceSearch=setTimeout(()=>runSearch(value), 400);
});
input.addEventListener('keydown', e=>{
  if(!currentResults.length) return;
  if(e.key==='ArrowDown'){ e.preventDefault(); setActive(Math.min(activeIndex+1, currentResults.length-1)); }
  else if(e.key==='ArrowUp'){ e.preventDefault(); setActive(Math.max(activeIndex-1,0)); }
  else if(e.key==='Enter' && activeIndex>=0){ choose(activeIndex); }
});
resetBtn.addEventListener('click', ()=>{
  try{ localStorage.removeItem(STORAGE_KEY); }catch(e){}
  NET=loadWeights();
  netstateEl.textContent='мережа готова';
  if(input.value.trim()) runSearch(input.value);
});
