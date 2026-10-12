"""The prosper landing page: hero, story, ladder, game browser, scope and footer.

Kept apart from the data pipeline in `gen_compat_site.py` so the generator stays about
parsing and joining, and this module stays about presentation. Everything dynamic comes in
as arguments; the template is a plain string with `@@TOKEN@@` placeholders (not an f-string),
so the CSS and the page script can use braces freely.

Copy here restates only what README.md and the ladder in prosper/docs/ROADMAP.md already
say. It makes no claim about any single title: per-title status comes from the trackers.
"""

from __future__ import annotations

import html

REPO_URL = "https://github.com/mattias800/prosper"

# What each rung means in plain words. The exact rung number stays in tooltips.
LADDER_COPY = {
    0: "Starts, but no game picture yet.",
    1: "First real pictures on screen.",
    2: "Title screen and menus.",
    3: "Gameplay renders on screen.",
    4: "A person looked and confirmed it looks right.",
    5: "Compared side by side with a real PS5.",
    6: "Finished: a reviewed automatic check guards it.",
}

SITE_CSS = """
:root{--bg:#06080d;--bg2:#0b0f17;--card:rgba(255,255,255,.04);--card2:rgba(255,255,255,.07);
--line:rgba(255,255,255,.09);--tx:#e9eef7;--mut:#8f9bb0;--a:#8b6cff;--b:#27d4f2;
--grad:linear-gradient(120deg,#8b6cff 0%,#27d4f2 100%);--r:16px}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:72px}
body{margin:0;background:var(--bg);color:var(--tx);line-height:1.55;
font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
-webkit-font-smoothing:antialiased}
a{color:inherit;text-decoration:none}img{max-width:100%}
a,button,select,summary,.btn,.chip,.ms,label[for]{cursor:pointer}
input[type=search],input[type=text]{cursor:text}
.shot,.shot img,.mosaic img,.pill{cursor:default}
button,.btn,.chip{-webkit-tap-highlight-color:transparent;user-select:none}
.wrap{max-width:1180px;margin:0 auto;padding:0 24px}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.85em;
background:var(--card2);padding:1px 6px;border-radius:6px}
nav.top{position:sticky;top:0;z-index:20;backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
background:rgba(6,8,13,.72);border-bottom:1px solid var(--line)}
nav.top .wrap{display:flex;align-items:center;gap:20px;height:64px}
.brand{display:flex;align-items:center;gap:10px;font-weight:700;font-size:18px;letter-spacing:-.01em}
img.logo{height:34px;width:auto;display:block}
.navlinks{display:flex;gap:22px;margin-left:auto;font-size:14px;color:var(--mut)}
.navlinks a:hover{color:var(--tx)}
.btn{display:inline-flex;align-items:center;gap:8px;padding:11px 20px;border-radius:999px;font-weight:600;
font-size:15px;border:1px solid var(--line);background:var(--card);transition:.2s}
.btn:hover{background:var(--card2);transform:translateY(-1px)}
.btn.primary{background:var(--grad);color:#06080d;border-color:transparent;
box-shadow:0 8px 30px rgba(139,108,255,.35)}
.btn.primary:hover{box-shadow:0 10px 38px rgba(39,212,242,.4)}
.btn.sm{padding:7px 15px;font-size:13px}
.hero{position:relative;overflow:hidden;padding:84px 0 56px;
background:radial-gradient(900px 480px at 15% -10%,rgba(139,108,255,.32),transparent 60%),
radial-gradient(800px 460px at 92% 8%,rgba(39,212,242,.22),transparent 60%)}
.hero .wrap{position:relative}
.eyebrow{display:inline-block;font-size:13px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
color:var(--b);padding:6px 12px;border:1px solid rgba(39,212,242,.35);border-radius:999px;
background:rgba(39,212,242,.08)}
h1.display{font-size:clamp(38px,6.4vw,76px);line-height:1.03;letter-spacing:-.035em;margin:20px 0 18px;
max-width:15ch}
.grad{background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent}
.lede{font-size:clamp(17px,2vw,21px);color:#c5cfdf;max-width:62ch;margin:0 0 28px}
.cta{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:12px}
.fine{font-size:13px;color:var(--mut);margin:0}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:44px 0 0}
.stat{background:var(--card);border:1px solid var(--line);border-radius:var(--r);padding:18px 20px}
.stat b{display:block;font-size:clamp(28px,4vw,40px);letter-spacing:-.03em;line-height:1.1}
.stat span{font-size:13px;color:var(--mut)}
.mosaic{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-top:36px}
.mosaic figure{margin:0;border-radius:12px;overflow:hidden;border:1px solid var(--line);aspect-ratio:16/10;
background:var(--bg2);transition:.25s;position:relative}
.mosaic figure:nth-child(odd){transform:translateY(10px)}
.mosaic figure:hover{transform:translateY(-4px) scale(1.03);z-index:2}
.mosaic img{width:100%;height:100%;object-fit:cover;display:block}
section{padding:84px 0}
section.alt{background:var(--bg2);border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
.kicker{font-size:13px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--a)}
h2{font-size:clamp(28px,4vw,44px);letter-spacing:-.03em;line-height:1.1;margin:10px 0 14px}
.sub{color:var(--mut);max-width:64ch;margin:0 0 40px;font-size:17px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:var(--r);padding:24px;transition:.25s}
.card:hover{background:var(--card2);transform:translateY(-3px);border-color:rgba(139,108,255,.45)}
.card .ico{width:42px;height:42px;border-radius:12px;display:grid;place-items:center;font-size:20px;
background:var(--grad);color:#06080d;margin-bottom:16px}
.card h3{margin:0 0 8px;font-size:18px;letter-spacing:-.01em}
.card p{margin:0;color:var(--mut);font-size:15px}
.ladder{display:grid;grid-template-columns:repeat(7,1fr);gap:10px}
.rung{background:var(--card);border:1px solid var(--line);border-radius:var(--r);padding:18px 14px;
position:relative;overflow:hidden}
.rung::before{content:"";position:absolute;left:0;right:0;top:0;height:4px;background:var(--c)}
.rung .n{font-size:12px;color:var(--mut);font-weight:600}
.rung h4{margin:6px 0 4px;font-size:17px}
.rung .c{font-size:28px;font-weight:800;letter-spacing:-.03em;color:var(--c)}
.rung p{margin:6px 0 0;font-size:13px;color:var(--mut)}
.toolbar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:12px}
.search{flex:1 1 260px;position:relative}
.search input{width:100%;padding:13px 16px 13px 42px}
.search::before{content:"\\1F50D";position:absolute;left:14px;top:50%;transform:translateY(-50%);font-size:14px;
opacity:.6;pointer-events:none}
input,select{background:var(--bg2);color:var(--tx);border:1px solid var(--line);border-radius:12px;
padding:12px 14px;font-size:15px;font-family:inherit}
input:focus,select:focus{outline:2px solid rgba(139,108,255,.6);border-color:transparent}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 22px}
.chip{cursor:pointer;border:1px solid var(--line);background:var(--card);color:var(--tx);border-radius:999px;
padding:7px 14px;font-size:14px;font-family:inherit;display:inline-flex;gap:8px;align-items:center}
.chip i{width:9px;height:9px;border-radius:50%;background:var(--c,#888);display:inline-block}
.chip em{font-style:normal;color:var(--mut);font-size:12px}
.chip:hover{background:var(--card2)}
.chip.on{background:var(--tx);color:#06080d;border-color:var(--tx)}
.chip.on em{color:#445}
.count{color:var(--mut);font-size:14px;margin:0 0 14px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:16px}
.game{background:var(--card);border:1px solid var(--line);border-radius:var(--r);overflow:hidden;
display:flex;flex-direction:column;transition:.25s}
.game:hover{border-color:rgba(139,108,255,.5);transform:translateY(-3px);background:var(--card2)}
.shot{aspect-ratio:16/9;background:linear-gradient(135deg,#1a1f33,#0e1220);position:relative;overflow:hidden}
.shot img{width:100%;height:100%;object-fit:cover;display:block}
.shot .ph{position:absolute;inset:0;display:grid;place-items:center;font-size:34px;font-weight:800;
color:rgba(255,255,255,.18);letter-spacing:-.02em}
.pill{position:absolute;top:10px;left:10px;padding:4px 11px;border-radius:999px;font-size:12px;font-weight:700;
color:#06080d;white-space:nowrap;box-shadow:0 4px 14px rgba(0,0,0,.4)}
.gbody{padding:16px 18px 18px;display:flex;flex-direction:column;gap:6px;flex:1}
.gbody h3{margin:0;font-size:17px;letter-spacing:-.01em}
.meta{font-size:12px;color:var(--mut)}
.ms{font-size:14px;color:#c0c9d8;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;
cursor:pointer;margin:4px 0 0}
.ms.open{-webkit-line-clamp:unset;display:block}
.glinks{margin-top:auto;padding-top:10px;font-size:13px;display:flex;gap:14px;color:var(--b)}
.glinks a:hover{text-decoration:underline}
.empty{padding:40px;text-align:center;color:var(--mut);border:1px dashed var(--line);border-radius:var(--r)}
.three{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}
.note{margin-top:28px;padding:18px 22px;border-radius:var(--r);border:1px solid rgba(255,196,87,.35);
background:rgba(255,196,87,.07);color:#f1d9a5;font-size:15px}
footer{border-top:1px solid var(--line);padding:44px 0 56px;color:var(--mut);font-size:14px}
footer .cols{display:flex;gap:28px;flex-wrap:wrap;justify-content:space-between;align-items:flex-start}
footer a{color:var(--tx)}footer a:hover{text-decoration:underline}
footer .links{display:flex;gap:20px;flex-wrap:wrap}
@media (max-width:980px){.ladder{grid-template-columns:repeat(4,1fr)}.mosaic{grid-template-columns:repeat(3,1fr)}
.stats{grid-template-columns:repeat(2,1fr)}}
@media (max-width:640px){.wrap{padding:0 16px}.navlinks a:not(.keep){display:none}.navlinks{gap:12px}
.hero{padding:48px 0 36px}section{padding:56px 0}.ladder{grid-template-columns:repeat(2,1fr)}
.mosaic{grid-template-columns:repeat(2,1fr)}.mosaic figure:nth-child(odd){transform:none}
.grid{grid-template-columns:1fr}.toolbar select{flex:1 1 45%}.btn{flex:1 1 auto;justify-content:center}
.brand span{display:none}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}html{scroll-behavior:auto}}
"""

PAGE_SCRIPT = """
var T=[],B=[],on="",q="",eng="",sort="rung";
var $=function(id){return document.getElementById(id);};
function e(s){return String(s).replace(/[&<>"']/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c];});}
function ini(t){return t.split(/[^A-Za-z0-9]+/).filter(Boolean).slice(0,2).map(function(w){return w[0];}).join("").toUpperCase();}
function chips(){
 var h='<button class="chip'+(on===""?' on':'')+'" data-s=""><i></i>All <em>'+T.length+'</em></button>';
 B.forEach(function(b){h+='<button class="chip'+(on===b.label?' on':'')+'" data-s="'+e(b.label)+'" style="--c:'+b.colour+'"><i></i>'+e(b.label)+' <em>'+b.count+'</em></button>';});
 $("chips").innerHTML=h;
}
function go(){
 var R=T.filter(function(t){
  return (!q||t.title.toLowerCase().indexOf(q)>-1||t.id.toLowerCase().indexOf(q)>-1||t.engine.toLowerCase().indexOf(q)>-1)
   &&(!on||t.status===on)&&(!eng||t.engine===eng);});
 R.sort(sort==="az"?function(a,b){return a.title.localeCompare(b.title);}
  :function(a,b){return ((b.rung==null?-1:b.rung)-(a.rung==null?-1:a.rung))||a.title.localeCompare(b.title);});
 $("count").textContent=R.length+" of "+T.length+" games";
 if(!R.length){$("grid").innerHTML='<div class="empty">No games match. Try clearing a filter.</div>';return;}
 $("grid").innerHTML=R.map(function(t){
  var img=t.cover?'<img loading="lazy" src="'+e(t.cover)+'" alt="'+e(t.title)+'">':'<div class="ph">'+e(ini(t.title))+'</div>';
  var doc=t.doc_url?'<a href="'+e(t.doc_url)+'">Status doc</a>':'';
  return '<article class="game"><div class="shot">'+img+'<span class="pill" style="background:'+t.colour+'" title="'+e(t.hint)+'">'+e(t.status)+'</span></div>'
   +'<div class="gbody"><h3>'+e(t.title)+'</h3><div class="meta">'+e(t.engine==="-"?"Engine unknown":t.engine)+' &middot; <code>'+e(t.id)+'</code></div>'
   +'<p class="ms" title="Click to expand">'+e(t.milestone==="-"?"No milestone written yet.":t.milestone)+'</p>'
   +'<div class="glinks"><a href="'+e(t.tracker.url)+'">Tracker</a>'+doc+'</div></div></article>';}).join("");
}
fetch("compat.json").then(function(r){return r.json();}).then(function(d){
 T=d.titles;B=d.buckets;
 var es={};T.forEach(function(t){if(t.engine&&t.engine!=="-")es[t.engine]=1;});
 $("eng").innerHTML='<option value="">All engines</option>'+Object.keys(es).sort().map(function(x){return '<option>'+e(x)+'</option>';}).join("");
 chips();go();
}).catch(function(){$("grid").innerHTML='<div class="empty">Could not load the game list. See the <a href="classic.html">static list</a>.</div>';});
$("q").addEventListener("input",function(ev){q=ev.target.value.toLowerCase();go();});
$("eng").addEventListener("change",function(ev){eng=ev.target.value;go();});
$("sort").addEventListener("change",function(ev){sort=ev.target.value;go();});
$("chips").addEventListener("click",function(ev){var b=ev.target.closest(".chip");if(!b)return;on=b.getAttribute("data-s");chips();go();});
$("grid").addEventListener("click",function(ev){var m=ev.target.closest(".ms");if(m)m.classList.toggle("open");});
"""

TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>prosper - run PS5 games natively on your PC</title>
<meta name="description" content="prosper is a PS5 to PC compatibility layer for Linux and Windows. Guest code runs natively, system libraries are reimplemented, and the GPU stream is translated to Vulkan.">
<meta name="theme-color" content="#06080d">
<meta property="og:title" content="prosper - run PS5 games natively on your PC">
<meta property="og:description" content="Wine/Proton for PS5. Live compatibility list for @@COUNT@@ titles.">
<link rel="icon" href="@@LOGO@@">
<style>@@CSS@@</style></head><body>
<nav class="top"><div class="wrap">
<a class="brand" href="#top"><img class="logo" src="@@LOGO@@" alt="prosper logo"><span>prosper</span></a>
<div class="navlinks"><a href="#how">How it works</a><a href="#progress">Progress</a><a href="#games">Games</a>
<a href="#scope">Scope</a><a class="btn sm keep" href="@@REPO@@">GitHub</a></div>
</div></nav>

<header class="hero" id="top"><div class="wrap">
<span class="eyebrow">Wine / Proton for PS5</span>
<h1 class="display">Run PS5 games <span class="grad">natively</span> on your PC.</h1>
<p class="lede">prosper is a user-space compatibility layer for Linux and Windows. The PS5 CPU is x86-64, so game code
runs natively with no CPU emulation. The work is loading the game, reimplementing Sony's system libraries, and
translating the GPU to Vulkan.</p>
<div class="cta"><a class="btn primary" href="#games">Browse @@COUNT@@ games</a>
<a class="btn" href="@@REPO@@">View source</a><a class="btn" href="@@REPO@@/blob/main/BUILDING.md">Build it</a></div>
<p class="fine">Experimental. Not a product: no support, no release schedule, and nothing works beyond the route documented for each title.</p>
<div class="stats">@@STATS@@</div>
<div class="mosaic">@@MOSAIC@@</div>
</div></header>

<section id="how"><div class="wrap">
<div class="kicker">How it works</div><h2>Native code, translated graphics</h2>
<p class="sub">No emulated CPU means no emulation overhead on the game itself. prosper only translates what the PC
genuinely cannot run.</p>
<div class="cards">
<div class="card"><div class="ico">&#9889;</div><h3>Runs natively</h3><p>Zen 2 is x86-64. Game code executes at host speed, so the game itself is what gets tested, not a second copy of it.</p></div>
<div class="card"><div class="ico">&#128230;</div><h3>Loader and linker</h3><p>SELF/ELF modules are parsed, relocated and linked into one address space against a global export table.</p></div>
<div class="card"><div class="ico">&#128218;</div><h3>Reimplemented libraries</h3><p>libc, libkernel, graphics and services are rebuilt from published interfaces, the way Wine reimplements Win32.</p></div>
<div class="card"><div class="ico">&#127912;</div><h3>GPU to Vulkan</h3><p>AGC command streams are decoded and RDNA2 shaders are recompiled to SPIR-V for a Vulkan 1.4 backend.</p></div>
</div></div></section>

<section class="alt" id="progress"><div class="wrap">
<div class="kicker">Progress</div><h2>Every game climbs the same ladder</h2>
<p class="sub">A running loop over a black screen is not gameplay. Each title is placed on one honest scale, from
"it starts" to "a reviewed check protects it". Counts below are live.</p>
<div class="ladder">@@LADDER@@</div>
</div></section>

<section id="games"><div class="wrap">
<div class="kicker">Games</div><h2>Compatibility list</h2>
<p class="sub">Search by name, ID or engine. Click a description to read the full note. Each card links to its
tracker issue, which is authoritative over this page.</p>
<div class="toolbar">
<div class="search"><input id="q" type="search" placeholder="Search games, PPSA ID or engine" aria-label="Search games"></div>
<select id="eng" aria-label="Engine"><option value="">All engines</option></select>
<select id="sort" aria-label="Sort"><option value="rung">Furthest first</option><option value="az">A to Z</option></select>
</div>
<div class="chips" id="chips"></div>
<p class="count" id="count"></p>
<div class="grid" id="grid"><div class="empty">Loading games&hellip;</div></div>
<noscript><p class="note">This list needs JavaScript. See the <a href="classic.html">static list</a>.</p></noscript>
</div></section>

<section class="alt" id="scope"><div class="wrap">
<div class="kicker">Scope</div><h2>What prosper does, and does not do</h2>
<p class="sub">Interoperability and preservation. You bring your own legally obtained dump.</p>
<div class="three">
<div class="card"><h3>No proprietary code</h3><p>No game files, keys, firmware or Sony code live in the repository.</p></div>
<div class="card"><h3>No decryption</h3><p>prosper holds no console keys and works only on module segments that are already unencrypted.</p></div>
<div class="card"><h3>Clean-room</h3><p>Library interfaces are reimplemented from published symbol and NID data. Third-party replacement libraries from dumps are refused.</p></div>
</div>
<div class="note">Independent project, not affiliated with Sony Interactive Entertainment. Status here describes specific
user-supplied dumps tested mainly on Linux; a different revision of a title may behave differently.</div>
</div></section>

<footer><div class="wrap"><div class="cols">
<div><b style="color:var(--tx)">prosper</b><br>PS5 to PC compatibility layer.<br>Page generated @@GENERATED@@ from the trackers.</div>
<div class="links"><a href="@@REPO@@">GitHub</a><a href="@@REPO@@/blob/main/COMPATIBILITY.md">COMPATIBILITY.md</a>
<a href="@@REPO@@/blob/main/BLOG.md">Progress blog</a><a href="@@REPO@@/blob/main/CONTRIBUTING.md">Contribute</a>
<a href="classic.html">Static list</a><a href="compat.json">compat.json</a><a href="llms.txt">llms.txt</a></div>
</div></div></footer>
<script>@@SCRIPT@@</script></body></html>
"""


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def _stat(value: int, label: str) -> str:
    return f'<div class="stat"><b>{value}</b><span>{_esc(label)}</span></div>'


def render_site(
    titles: list[dict],
    stats: list[dict],
    ladder: list[dict],
    generated: str,
    logo: str,
) -> str:
    """Assemble the landing page. `ladder` rows: rung,label,colour,hint,count."""
    total = len(titles)
    complete = sum(1 for t in titles if t["rung"] == 6)
    playable = sum(1 for t in titles if t["rung"] is not None and t["rung"] >= 3)
    menus = sum(1 for t in titles if t["rung"] is not None and t["rung"] >= 2)
    stat_html = "".join(
        [
            _stat(total, "games tracked"),
            _stat(menus, "reach menus or better"),
            _stat(playable, "render gameplay"),
            _stat(complete, "complete"),
        ]
    )
    rungs = "".join(
        f'<div class="rung" style="--c:{r["colour"]}" title="{_esc(r["hint"])}">'
        f'<div class="n">RUNG {r["rung"]}</div><h4>{_esc(r["label"])}</h4>'
        f'<div class="c">{r["count"]}</div><p>{_esc(LADDER_COPY[r["rung"]])}</p></div>'
        for r in ladder
    )
    # Furthest-along titles first: they are the ones with a clear, bright game picture.
    shots = sorted(
        (t for t in titles if t["cover"] and t["rung"] is not None and t["rung"] >= 3),
        key=lambda t: (-t["rung"], t["title"].lower()),
    )[:6]
    mosaic = "".join(
        f'<figure><img loading="lazy" src="{_esc(t["cover"])}" alt="{_esc(t["title"])}"></figure>'
        for t in shots
    )
    page = TEMPLATE
    for token, value in {
        "@@CSS@@": SITE_CSS,
        "@@SCRIPT@@": PAGE_SCRIPT,
        "@@STATS@@": stat_html,
        "@@LADDER@@": rungs,
        "@@MOSAIC@@": mosaic,
        "@@LOGO@@": _esc(logo),
        "@@REPO@@": REPO_URL,
        "@@COUNT@@": str(total),
        "@@GENERATED@@": _esc(generated),
    }.items():
        page = page.replace(token, value)
    return page
