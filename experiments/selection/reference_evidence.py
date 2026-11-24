"""Bind README examples to training columns and extract declared evaluation measures from cited papers."""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import unicodedata
from tempfile import NamedTemporaryFile
from urllib.request import HTTPRedirectHandler, build_opener, Request


def example_scope(text, frame):
    """A label/example table can identify a subtask even when its directory and columns have been renamed."""
    lines=text.splitlines()
    headings=[i for i,s in enumerate(lines) if re.match(r'^#{1,2}\s+',s)]
    boundaries=sorted({0,*headings,len(lines)})
    vocabulary={col:set(frame[col]) for col in frame}
    matches=[]
    for begin,end in zip(boundaries,boundaries[1:]):
        for i in range(begin,end-1):
            header=lines[i].strip()
            if not header.startswith('|') or not re.fullmatch(r'[\s|:\-]+',lines[i+1]) or '-' not in lines[i+1]:
                continue
            cells=[s.strip().strip('*` ').casefold() for s in header.strip('|').split('|')]
            roles=[j for j,c in enumerate(cells) if c in {'intent','class','label','category','target','类别','标签','意图'}]
            if len(roles)!=1:
                continue
            values=set()
            for s in lines[i+2:end]:
                if not s.strip().startswith('|'):
                    break
                row=[v.strip().strip('`') for v in s.strip().strip('|').split('|')]
                if len(row)==len(cells) and row[roles[0]]:
                    values.add(row[roles[0]])
            if len(values)<2:
                continue
            fields=[col for col,vocab in vocabulary.items() if values<=vocab]
            if len(fields)>1:
                raise ValueError('README example labels match multiple training fields')
            if fields:
                matches.append((begin,end,fields[0],i+1,sorted(values)))
    if not matches:
        return text,None,None
    if len({m[:3] for m in matches})!=1:
        raise ValueError('README example tables bind more than one task or target')
    begin,end,col,line,examples=matches[0]
    # Pad the beginning so subsequent citations keep their original line numbers.
    scoped='\n'*begin+'\n'.join(lines[begin:end])
    return scoped,col,dict(field='target',source='README example labels matched to training vocabulary',
        line=line,scope=lines[begin],scope_start=begin+1,scope_end=end,value=col,examples=examples)


def paper_url(link):
    if re.fullmatch(r'https://arxiv\.org/(?:abs|pdf)/[0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?(?:\.pdf)?/?',link):
        paper=link.rstrip('/').rsplit('/',1)[1].removesuffix('.pdf')
        return f'https://arxiv.org/pdf/{paper}'
    if re.fullmatch(r'https://aclanthology\.org/(?:[0-9]{4}\.[A-Za-z0-9_-]+\.[0-9]+|[A-Z][0-9]{2}-[0-9]{4})(?:\.pdf|/)?',link):
        return link.rstrip('/').removesuffix('.pdf')+'.pdf'
    return None


class PaperRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not paper_url(newurl):
            raise ValueError('Reference redirect is not an academic PDF URL')
        return super().redirect_request(req,fp,code,msg,headers,newurl)


def atomic_bytes(path, data):
    with NamedTemporaryFile(dir=path.parent,delete=False) as tmp:
        tmp.write(data)
        name=tmp.name
    os.replace(name,path)


def load_paper(url, cache, fetch):
    if not (fetch or cache):
        raise ValueError('Cited-paper metric requires a cached reference or enabled reference retrieval')
    key=hashlib.sha256(url.encode()).hexdigest()
    if cache:
        pdf=cache/(key+'.pdf')
        metadata=cache/(key+'.json')
        if pdf.is_file() and metadata.is_file():
            data=pdf.read_bytes()
            expected=json.loads(metadata.read_text(encoding='utf-8'))
            if expected.get('url')!=url or expected.get('sha256')!=hashlib.sha256(data).hexdigest():
                raise ValueError('Reference cache fingerprint mismatch')
            return data,'cache'
    if not fetch:
        raise ValueError('Cited paper is absent from the offline cache')
    with build_opener(PaperRedirect()).open(Request(url,headers={'User-Agent':'AIAA3102 academic-reference reader'}),timeout=25) as response:
        data=response.read(12*1024*1024+1)
    if not data.startswith(b'%PDF') or len(data)>12*1024*1024:
        raise ValueError('Cited document is not a supported PDF')
    if cache:
        cache.mkdir(parents=True,exist_ok=True)
        atomic_bytes(pdf,data)
        atomic_bytes(metadata,json.dumps({'url':url,'sha256':hashlib.sha256(data).hexdigest()},indent=2).encode())
    return data,'download'


def paper_claims(pages, patterns):
    # Secondary mentions, results tables and metrics for other tasks are not declarations.
    declarations=(r'\b(?:we|the authors)\s+(?:report|use|adopt|employ)\s+(.{1,80}?)\s+as\s+(?:the|our)\s+(?:main|primary|official)\s+(?:evaluation\s+)?(?:measure|metric|criterion)\b',
                  r'\b(?:our|the)\s+(?:main|primary|official)\s+(?:evaluation\s+)?(?:metric|measure|criterion)\s+is\s+(.{1,80}?)(?=\s+for\b|[.;])')
    claims=[]
    for page,text in enumerate(pages,1):
        flat=' '.join(text.split())
        for sentence in re.split(r'(?<=[.!?])\s+',flat):
            for pattern in declarations:
                m=re.search(pattern,sentence,re.I)
                if not m:
                    continue
                term=m[1].strip()
                names=[key for key,value in patterns.items() if re.fullmatch(value,term,re.I)]
                universal=bool(re.search(r'\b(?:for|across|in|on)\s+all\s+(?:(?:our|the|experimental|evaluation)\s+)*(?:runs|tasks|datasets|data sets|experiments|benchmarks)\b',sentence,re.I))
                if re.search(r'\b(?:except|excluding|other than|unless|but not)\b|\bnot\s+for\s+all\b',sentence,re.I):
                    universal=False
                claims.append(dict(page=page,term=term,metric=names[0] if len(names)==1 else None,
                                   universal=universal,text=sentence.strip()[:550]))
    return claims


def benchmark_measures(pages, task, patterns):
    """Corroborate a full benchmark comparison with a separately labelled figure."""
    identifier=re.escape(task)
    table=[];figure=[]
    measures={**patterns,"accuracy":r"accuracy|acc\.?"}
    def metric_names(line):
        rest=line
        found=set()
        for key,pattern in measures.items():
            regex=r"(?<!\w)(?:"+pattern+r")(?!\w)"
            if re.search(regex,rest,re.I):
                found.add(key)
                rest=re.sub(regex," ",rest,flags=re.I)
        return found,rest
    for page,text in enumerate(pages,1):
        lines=unicodedata.normalize("NFKC",text).splitlines()
        for i,s in enumerate(lines):
            if re.match(r"^\s*Model\s+(?:Dev|Val(?:idation)?)\b",s,re.I) and re.search(r"\btest\b",s,re.I):
                keys,rest=metric_names(s)
                if re.sub(r"\b(?:model|dev|val|validation|test)\b|[\s|./:%()-]","",rest,flags=re.I):
                    continue
                body=lines[i+1:i+17]
                caption=next((j for j,v in enumerate(body) if re.match(r"Table\s+\d+[:.]",v,re.I)),None)
                if caption is None or sum(bool(re.search(r"\d+(?:\.\d+)?\s+\d",v)) for v in body[:caption])<2:
                    continue
                caption_text=" ".join(body[caption:caption+2])
                if re.search(r"\bresults\s+on\s+(?:the\s+)?"+identifier+r"\b",caption_text,re.I) and not re.search(r"\bablation|\bsubset|\bauxiliary",caption_text,re.I):
                    table.extend(dict(metric=k,page=page,text=s,caption=body[caption]) for k in keys)
            if re.match(r"Figure\s+\d+:",s,re.I):
                caption=" ".join(lines[i:i+3]).split(". ",1)[0]
                if re.search(r"\bfor\s+(?:various|different|multiple)\s+models\s+on\s+(?:the\s+)?"+identifier+r"\s+(?:dev|validation|test)\s+set\b",caption,re.I):
                    keys,_=metric_names(caption.split(" for ",1)[0])
                    figure.extend(dict(metric=k,page=page,text=caption) for k in keys)
    keys={e["metric"] for e in table+figure}
    if table and figure and len(keys)==1:
        return dict(metric=keys.pop(),table=table[0],figure=figure[0])


def cited_metric(scoped, cache, fetch, patterns, audit):
    refs={}
    for n,line in enumerate(scoped.splitlines(),1):
        applies=re.search(r'(?:\b(?:cite|reference)\b.*\[|\b(?:evalua\w*|metric|scoring)\b.*\[|\b(?:original|dataset)\s+paper\b.*\[|评测|评分|原论文)',line,re.I)
        applies=applies or re.search(r'\[our (?:dataset )?paper\]\(',line,re.I)
        if not applies or re.search(r'\b(?:baseline|related work|further reading|unrelated)\b',line,re.I):
            continue
        for match in re.finditer(r'\[([^\]\n]+)\]\((https://[^)\s]+)\)',line):
            url=paper_url(match[2])
            if url:
                refs.setdefault(url,(match[2],n,match[1]))
    if len(refs)!=1:
        raise ValueError(f'Expected one scoped evaluation metric or uniquely cited paper; found {len(refs)} papers')
    url,(link,line,title)=next(iter(refs.items()))
    report=dict(source='cited_paper',readme_link=link,readme_line=line,citation_title=title,url=url)
    audit.setdefault('references',[]).append(report)
    data,origin=load_paper(url,cache,fetch)
    report.update(access=origin,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    try:
        from pypdf import PdfReader
        reader=PdfReader(io.BytesIO(data))
        if len(reader.pages)>80:
            raise ValueError('Reference exceeds the page limit')
        pages=[p.extract_text() or '' for p in reader.pages]
    except ImportError as exc:
        raise ValueError('PDF metric extraction requires pypdf') from exc
    except Exception as exc:
        raise ValueError(f'Could not read cited PDF: {type(exc).__name__}') from exc
    def normal(s):
        return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',s).casefold()))
    if normal(title) in {"our paper","our dataset paper"}:
        titles=re.findall(r"(?im)^\s*title\s*=\s*\{([^{}\n]+)\}",scoped)
        if len(titles)!=1:
            raise ValueError("Cannot verify the cited paper title from the bibliography")
        title=titles[0]
        report["citation_title"]=title
    if len(normal(title).split())<3 or normal(title).replace(" ","") not in normal(pages[0][:2500]).replace(" ",""):
        raise ValueError('Cannot verify the cited paper title against its first page')
    binding=next((e for e in audit['evidence'] if 'scope_start' in e),None)
    if binding:
        tokens=set(normal(binding['scope']).split())-{'the','data','set','dataset','task','classification','examples','overview'}
        full=normal(' '.join(pages))
        names=[t for t in sorted(tokens) if len(t)>=3 and re.search(r'\b'+re.escape(t)+r'\d*\b',full)]
        if not names:
            raise ValueError('Cited paper does not corroborate the bound task section')
        report['matched_scope_tokens']=names
    report['title_verified']=True
    claims=paper_claims(pages,patterns)
    report['claims']=claims
    metrics={c['metric'] for c in claims}
    if not claims and not binding:
        names=re.findall(r"(?im)^#\s+([\w][\w -]*?)\s+dataset\s*$",scoped)
        if len(names)==1 and (" "+normal(names[0])+" ") in (" "+normal(title)+" "):
            supported=benchmark_measures(pages,names[0],patterns)
            if supported:
                audit["evidence"].append(dict(field="metric",source="cited benchmark comparison and figure",url=url,
                                              sha256=report["sha256"],**supported))
                return supported["metric"]
    if len(metrics)!=1 or None in metrics or not any(c['universal'] for c in claims):
        raise ValueError('Cited paper lacks a unique main metric explicitly applicable across experiments')
    winner=next(c for c in claims if c['universal'])
    audit['evidence'].append(dict(field='metric',source='cited_paper',url=url,sha256=report['sha256'],**winner))
    return winner['metric']
