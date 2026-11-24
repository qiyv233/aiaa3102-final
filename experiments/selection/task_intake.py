"""Build a classification manifest from stated task semantics and verified training schema."""
import argparse
import csv
import fnmatch
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]/"Agent"))
from task_data import data_file, data_root, jsonl_frame
from reference_evidence import example_scope, cited_metric
from objective import METRICS as TRAINABLE


class IntakeError(ValueError):
    pass


class MissingMetric(IntakeError):
    pass


def require(condition, message):
    if not condition:
        raise IntakeError(message)


def readmes(source):
    if source.is_dir():
        paths=(p.relative_to(source).as_posix() for p in source.rglob("*") if p.name.casefold() in {"readme.md","readme.txt","readme"})
    else:
        with ZipFile(source) as archive:
            paths=archive.namelist()
    return sorted(p for p in paths if PurePosixPath(p).name.casefold() in {"readme","readme.md","readme.txt"}
                  and not any(i.startswith(".") or i in {"__MACOSX",".."} for i in PurePosixPath(p).parts)
                  and not PurePosixPath(p).is_absolute())


def blocks(text):
    lines=text.splitlines()
    result=[]
    i=0
    while i<len(lines):
        first=i+1
        line=lines[i].strip().replace("**","")
        i+=1
        heading=re.fullmatch(r"(?:#+\s*)?(inputs?|features?|predictors?|columns?|target(?: column)?|(?:primary |evaluation )?metric|输入字段|目标列|评分指标)\s*[:：]?",line,re.I)
        if heading:
            values=[]
            while i<len(lines) and (item:=re.fullmatch(r"\s*[-*+]\s+(.+)",lines[i])):
                values.append(item[1].strip())
                i+=1
            if not values and i<len(lines) and lines[i].strip() and not lines[i].lstrip().startswith("#"):
                values.append(lines[i].strip().replace("**",""))
                i+=1
            if values:
                line=f"{heading[1]}: "+", ".join(values)
        for sentence in re.split(r"(?<=[.!?])\s+|(?<=[。！？])|[;；]",line):
            if sentence.strip():
                result.append((first,sentence.strip()))
    return result


def quoted(s):
    return re.findall(r"`([^`]+)`",s)


def listed(s):
    # Keep quoted names intact, including spaces, commas, and conjunctions.
    parts=re.findall(r"\x60[^\x60]+\x60|[^,\x60、，]+",s)
    fields=[]
    for part in parts:
        if part.startswith(chr(96)):
            fields.append(part[1:-1])
            continue
        for field in re.split(r"\band\b|和",part,flags=re.I):
            field=field.strip(" .*，、。'\"")
            if field:
                fields.append(field)
    return fields


def single(values, description):
    unique=list(dict.fromkeys(values))
    require(len(unique)==1,f"Expected one {description}, found {unique}")
    return unique[0]


METRICS={"balanced_accuracy":r"balanced[ _-]+accuracy|平衡准确率",
         "macro_f1":r"macro[ _-]*f1(?:[ -]score)?|宏平均\s*F1",
         "weighted_f1":r"weighted[ _-]*f1|加权\s*F1",
         "micro_f1":r"micro[ _-]*f1|微平均\s*F1",
         "average_precision":r"\baverage[ _-]+precision\b",
         "roc_auc":r"(?:ROC[ _-]*)?AUC\b", "log_loss":r"log[ _-]*loss|对数损失",
         "rmse":r"\bRMSE\b|root mean squared error|均方根误差",
         "mae":r"\bMAE\b|mean absolute error|平均绝对误差",
         "precision":r"\bprecision\b|精确率", "recall":r"\brecall\b|召回率",
         "f1":r"\bF1(?:[ -]score)?\b", "accuracy":r"\baccuracy\b|准确率|正确率"}


def metric_of(lines, evidence):
    mentions=[]
    primary=[]
    top_k=r"\btop[ _-]+(\d+|k)[ _-]+accuracy\b"
    for line,s in lines:
        if not re.search(r"\b(?:metric|score|report|evaluat\w*|assess\w*|measur\w*|optim\w*|accuracy|AUC|F1)\b|指标|准确率|正确率|评估|报告",re.sub(r"\x60[^\x60]+\x60"," ",s),re.I):
            continue
        for part in re.split(r"\bbut\b|,\s*(?=secondary\b)|(?=\balso\b|另报|同时报告)",s,flags=re.I):
            # Exclude fields and filenames from metric matching.
            rest=re.sub(r"\x60([^\x60]+)\x60",lambda m: m[1] if any(re.fullmatch(p,m[1],re.I) for p in (*METRICS.values(),top_k)) else " ",part)
            rest=re.sub(r"\b(?:rather than|instead of|do not (?:use|report)|don't (?:use|report)|not evaluated (?:on|by))\b.*|不(?:采用|使用|报告).*"," ",rest,flags=re.I)
            found=[f"top_{k.lower()}_accuracy" for k in re.findall(top_k,rest,re.I)]
            rest=re.sub(top_k," ",rest,flags=re.I)
            for name,pattern in METRICS.items():
                if re.search(pattern,rest,re.I):
                    found.append(name)
                    rest=re.sub(pattern," ",rest,flags=re.I)
            primary_clause=re.search(r"\b(?:primary|official|main)\s+(?:evaluation\s+)?(?:metric|score)\b|主要指标|主指标",part,re.I)
            if found or primary_clause:
                evidence.append({"field":"metric","line":line,"text":part.strip(),"mentions":found})
            if primary_clause:
                require(found,"Unrecognised primary evaluation metric in README")
                primary.extend(found)
            if not re.search(r"\b(?:secondary|also)\b|次要|另报|同时报告",part,re.I):
                mentions.extend(found)
    if not (primary or mentions):
        raise MissingMetric("No primary evaluation metric in the supplied instructions")
    return single(primary or mentions,"evaluation metric in README")


def split_files(names, lines):
    aliases={"train":("train","training"),"validation":("val","valid","validation","dev"),"test":("test","testing")}
    choices={k:[] for k in aliases}
    for name in names:
        if Path(name).suffix.lower() not in (".csv",".npy",".jsonl",".ndjson"): continue
        tokens=set(re.split(r"[^a-z0-9]+",Path(name).stem.lower()))
        role=[k for k,opts in aliases.items() if tokens.intersection(opts)]
        require(len(role)<=1,f"Ambiguous split filename {name}")
        # Only honour a custom name when a clause explicitly pairs that single file and role.
        for _,part in lines:
            if quoted(part).count(name)!=1 or sum(v in names for v in quoted(part))!=1: continue
            prose=re.sub(r"\x60[^\x60]*\x60"," ",part)
            words=set(re.findall(r"[a-z]+",prose.lower()))
            roles=[k for k,opts in aliases.items() if words.intersection(opts)]
            if len(roles)==1:
                role.append(roles[0])
        if role:
            choices[single(role,f"split role of {name}")].append(name)
    require(choices["train"],"No identifiable training file")
    return choices


def array_roles(files,lines):
    # Stems such as X_train and labels_train are conventions. A README can declare other stems.
    def role(name):
        tokens=set(re.split(r"[^a-z0-9]+",Path(name).stem.lower()))
        found=[]
        for _,line in lines:
            for clause in re.split(r"[;；]",line):
                patterns=[s for s in quoted(clause) if s.lower().endswith(".npy")]
                if len(patterns)==1 and fnmatch.fnmatchcase(name,patterns[0]):
                    words=re.sub(r"\x60[^\x60]*\x60"," ",clause)
                    if re.search(r"\blabels?\b|\btargets?\b|标签",words,re.I): found.append("labels")
                    if re.search(r"\bfeatures?\b|特征",words,re.I): found.append("features")
        if tokens & {"y","label","labels","target","targets"}: found.append("labels")
        if tokens & {"x","c","feature","features","data"}: found.append("features")
        return single(found,f"array role of {name}")
    result={}
    for split,paths in files.items():
        if not paths: continue
        by_role={}
        for p in paths:
            require(p.endswith(".npy"),"Mixed CSV/NumPy split layout")
            kind=role(p)
            require(kind not in by_role,f"Multiple {kind} arrays for {split}")
            by_role[kind]=p
        require("features" in by_role and ("labels" in by_role or split=="test"),f"Missing {split} features or labels")
        result[split]=by_role["features"]
        if "labels" in by_role: result[split+"_labels"]=by_role["labels"]
    return result


def array_names(lines):
    candidates=[]
    for line,s in lines:
        if not re.search(r"\b(?:columns?|feature names)\b|列名|特征列",s,re.I): continue
        match=re.search(r"(?:\bare\s+|:\s*|：\s*|依次为\s*|分别为\s*)(.+)",s,re.I)
        if match:
            names=listed(match[1])
            if names: candidates.append((names,line,s))
    require(len(candidates)==1,"NumPy needs one ordered column list in the README")
    return candidates[0]


def field_exclusions(sentence, columns):
    verb = r"(?:\b(?:exclude|ignore|do not use)\s+|(?:忽略|排除)\s*)"
    match = re.search(verb, sentence, re.I)
    if not match:
        return []
    obj = sentence[match.end():].strip().rstrip(".。").strip()
    prefix = re.match(r"(?:(?:the\s+)?(?:columns?|fields?|features?|predictors?)\s*[:：]?\s+|(?:字段|列|特征)\s*[:：]?\s*)", obj, re.I)
    if prefix:
        obj = obj[prefix.end():]
    role = re.search(r"\s+(?:as|from)\s+(?:(?:an?|the|model)\s+)*(?:inputs?|features?|predictors?)(?:\s+(?:list|set|columns?))?$|\s*(?:作为|用作)(?:输入|特征)$", obj, re.I)
    if role:
        obj = obj[:role.start()]
    fields = listed(obj)
    # Plain narrative about excluded examples carries no schema authority.
    # Explicit column declarations are still checked for misspellings.
    quoted_names = set(quoted(obj))
    names_only = all(v in columns or v in quoted_names or re.fullmatch(r"[\w-]+",v) for v in fields)
    lead = sentence[:match.start()].strip(" \t*-•")
    direct = not lead or lead.casefold() in {"we","the model","our model","我们"}
    if fields and (prefix or role or direct and names_only and (quoted_names or any(v in columns for v in fields))):
        require(set(fields) <= set(columns), "Excluded fields missing from training header")
        return fields
    return []


def csv_roles(columns,lines,evidence,example_target=None,documented_inputs=None):
    targets=[]
    inputs=[]
    excluded=set()
    role=r"(?:\b(?:target|outcome|response|label)(?:\s+(?:column|variable|field))?\b|目标列|标签列)"
    token=r"(?:\x60([^\x60]+)\x60|['\"“]([^'\"”]+)['\"”]|([\w-]+))"
    declarations=[
        role+r"\s*(?:is\b|=|:|：|为|是|named\b)\s*"+token,
        role+r"\s+\x60([^\x60]+)\x60",
        r"\x60([^\x60]+)\x60\s+(?:is|as)\s+(?:the\s+)?(?:target|label|outcome)\b",
        r"\bpredict\s+(?:the\s+)?"+token+r"\s+(?:column\b|using\b|from\b)",
    ]
    for line,s in lines:
        hits=[]
        for pattern in declarations:
            for m in re.finditer(pattern,s,re.I):
                hits.append(next(value for value in m.groups() if value is not None))
        if hits:
            evidence.append({"field":"target","line":line,"text":s})
            targets.extend(hits)
        values=field_exclusions(s,columns)
        if values:
            excluded.update(values)
            evidence.append({"field":"excluded_features","line":line,"text":s,"value":values})
            continue
        m=re.search(r"\buse\s+(.+?)\s+as\s+(?:the\s+)?(?:input|feature|predictor)s?\b|"
                    r"(?:\b(?:inputs?|predictors?|features?)\s*(?:are\b|:|=)|输入字段为|输入列为|特征列为|输入字段[：:])\s*(.+)|"
                    r"\bpredict\b.+?\b(?:using|from)\s+(.+)",s,re.I)
        if m:
            fields=listed(next(g for g in m.groups() if g is not None))
            inputs.append(fields)
            evidence.append({"field":"features","line":line,"text":s})
    if example_target:
        targets.append(example_target)
    if targets:
        target=single(targets,"target declaration")
        require(target in columns,f"README target {target!r} is absent from the training header")
    else:
        target=single([c for c in columns if c.casefold() in {"target","label","class","y","标签"}],"unambiguous conventional training label")
        evidence.append({"field":"target","source":"training header","inferred":True,"value":target})
    if inputs:
        fields=list(single([tuple(v) for v in inputs],"explicit feature list"))
        require(not (set(fields)-set(columns)),f"Declared features missing from training: {set(fields)-set(columns)}")
        require(documented_inputs is None or set(fields)==set(documented_inputs),"Input declarations disagree")
    elif documented_inputs is not None:
        fields=list(documented_inputs)
    else:
        excluded.update(c for c in columns if c.casefold() in {"id","row_id","sample_id","record_id","index"})
        fields=[c for c in columns if c!=target and c not in excluded]
        evidence.append({"field":"features","source":"training header minus target and excluded fields","value":fields})
    require(fields and len(set(fields))==len(fields) and target not in fields and not set(fields)&excluded,
            "Feature roles conflict, overlap the label, or are empty")
    return target,fields


def classification(labels):
    labels=pd.Series(labels)
    require(not labels.isna().any() and labels.astype(str).str.strip().ne("").all(),"Missing training labels")
    counts=labels.value_counts()
    require(len(counts)>=2,"Training file has fewer than two classes")
    return len(counts)


def ordered_comparison(lines):
    return any(re.search(r"which of (?:the )?two .+?\b(?:higher|better|wins?|preferred|more)\b|pairwise (?:comparison|ranking)|两个.+(?:更|胜)",s,re.I) for _,s in lines)


def json_contract(frame, kinds, lines, evidence):
    """Bind a documented yes/no record or an explicit classification schema to actual fields."""
    columns = list(frame)
    target = inputs = None
    boolean_task = any(re.search(r"\byes\s*/\s*no\s+(?:questions?|answers?)\b|\bboolean\s+(?:answer|target|label)s?\b", s, re.I) for _,s in lines)
    tuples = []
    for number,s in lines:
        match = re.search(r"\b(?:tuple|triple|triplet)\s+(?:of\s+)?\(([^()]+)\)",s,re.I)
        if match and boolean_task:
            names = [part.strip(" `\t'\"") for part in match[1].split(",")]
            require(len(names)>=2 and len(set(names))==len(names) and set(names)<=set(columns),"Documented tuple disagrees with JSON keys")
            tuples.append(tuple(names))
            evidence.append(dict(field="record_tuple",line=number,value=names))
    if tuples:
        members = single(tuples,"documented record tuple")
        booleans = [k for k in members if kinds.get(k)=="boolean"]
        # With two Boolean fields the explicit target declaration must disambiguate.
        if len(booleans)==1:
            target=booleans[0]
        else:
            target,_=csv_roles(columns,lines,[],None)
            require(target in booleans,"No documented Boolean target matches the record tuple")
        inputs=[k for k in members if k!=target]
        evidence.append(dict(field="target",source="yes/no task and Boolean type in documented tuple",value=target))
    target,features=csv_roles(columns,lines,evidence,target,inputs)
    require(not frame[[target,*features]].isna().any().any(),"Required JSONL field absent or null in training")
    if boolean_task and tuples:
        require(kinds.get(target)=="boolean","Declared yes/no outcome conflicts with JSON type")
    from features import infer_modality
    text_pair=any(re.search(r"\btext[ -]pair classification\b|\bquestion answering\b",s,re.I) for _,s in lines)
    pair=ordered_comparison(lines)
    if pair:
        require(not boolean_task and len(features)==2 and frame[target].nunique()==2 and
                all(kinds.get(k)=="string" for k in features),"Ambiguous or unsupported ordered text comparison")
    mode="text" if pair or (text_pair and all(kinds.get(k)=="string" for k in features)) else infer_modality(frame[features])
    result=dict(target=target,features=features,modality=mode,native_format="jsonl",
                json_types={k:kinds[k] for k in [*features,target]})
    if pair:
        result["pairwise"]=True
    if mode=="table":
        result["numeric"]=[k for k in features if kinds[k]=="number"]
    declared=[]
    for number,s in lines:
        decl=re.search(r"(?:\bgroup(?:ed)?\s+by\s+|\bgroup(?:ing)?\s+field\s*[:=]\s*|分组字段[：:]\s*)[`'\"]?([\w-]+)",s,re.I)
        if decl:
            declared.append(decl[1])
            evidence.append(dict(field="group_field",line=number,value=decl[1],source="task instructions"))
    if declared:
        require(not pair and mode!="image","Explicit field conflicts with the required grouping strategy")
        result["group_field"]=single(declared,"group field")
        require(result["group_field"] in features,"Group field must refer to a declared input")
    elif tuples and text_pair:
        contexts=[k for k in features if k.casefold() in {"passage","context","document","evidence"}]
        if len(contexts)==1 and kinds[contexts[0]]=="string":
            result["group_field"]=contexts[0]
            evidence.append(dict(field="group_field",source="Agent grouping choice for documented evidence input",value=contexts[0]))
    return result


def resolve_metric(scoped,lines,locations,get_bytes,cache,fetch,audit):
    decisions=[]
    evidence=audit["evidence"]
    try:
        decisions.append(metric_of(lines,evidence))
    except MissingMetric:
        pass
    # Only scoring instructions are read. Data splits and executable scorers are never opened here.
    canonical={"evaluation.md","evaluation.txt","metrics.md","metrics.txt","scoring.md","scoring.txt"}
    linked={}
    for m in re.finditer(r"\[([^\]]*(?:metric|evaluation|scoring|评分|评价)[^\]]*)\]\(([^)]+)\)",scoped,re.I):
        name=m[2]
        if re.fullmatch(r"[\w.-]+\.(?:md|txt)",name,re.I) and not re.search(r"(?:^|[^a-z])(?:test|val|validation|train)(?:[^a-z]|$)",name,re.I):
            linked[name.casefold()]=name
    extra=[]
    positive_guides=[]
    for directory,names in locations:
        for n in sorted(names):
            if n.casefold() in canonical or n.casefold() in linked:
                path=(directory/n).as_posix()
                if path not in extra:
                    extra.append(path)
    for path in extra:
        guide=blocks(get_bytes(path).decode("utf-8-sig"))
        local=[]
        result=metric_of(guide,local)
        evidence.extend({**row,"source":path} for row in local)
        decisions.append(result)
        positive_guides.extend((number,s,path) for number,s in guide if re.search(r"positive\s+(?:class|label)|正类",s,re.I))
    if decisions:
        return single(decisions,"consistent evaluation metric across task documents"),positive_guides
    return cited_metric(scoped,cache,fetch,METRICS,audit),positive_guides


def inspect(source, chosen, trust_objects, audit, cache=None, fetch=False):
    available=readmes(source)
    if chosen is None: chosen=single(available,"task README (use --readme to select one)")
    require(chosen in available,f"Select a README inside the data root. Available: {available}")
    parent=PurePosixPath(chosen).parent
    evidence=audit["evidence"]
    def get_bytes(root,relative):
        raw=data_file(root,relative).read_bytes()
        audit["opened"].append({"file":relative,"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw)})
        return raw
    with data_root(source) as root:
        folder=data_file(root,str(parent)) if str(parent)!="." else root
        names=[p.name for p in folder.iterdir() if p.is_file() and not p.name.startswith(".")]
        text=get_bytes(root,chosen).decode("utf-8-sig")
        lines=blocks(text)
        locations=[(parent,names)]
        try:
            splits=split_files(names,lines)
        except IntakeError as exc:
            if str(exc)!="No identifiable training file":
                raise
            possible=[]
            for child in folder.iterdir():
                if not child.is_dir() or child.name.startswith(".") or child.name.casefold() in {"test","val","validation","dev","__macosx"}:
                    continue
                members=[p.name for p in child.iterdir() if p.is_file()]
                if any(re.search(r"(?:^|[_-])train(?:ing)?(?:[_-]|\.)",n,re.I) and n.lower().endswith((".csv",".npy",".jsonl",".ndjson")) for n in members):
                    possible.append((child.name,members))
            if not possible:
                from publisher_tsv import registered_task
                spec=registered_task(root,parent,names,get_bytes,audit)
                if spec is not None:
                    target,inputs=csv_roles(spec["tsv_columns"],lines,evidence,example_target=spec["target"])
                    require(inputs==spec["features"] and target==spec["target"],"README conflicts with registered TSV roles")
                    primary=[(n,s) for n,s in lines if re.search(r"\b(?:primary|official|main)\s+(?:evaluation\s+)?(?:metric|score)\b|主要指标|主指标",s,re.I)]
                    if primary:
                        require(metric_of(primary,evidence)==spec["metric"],"README conflicts with the registered scoring function")
                    audit["readme"]=chosen
                    return spec
            require(len(possible)==1,"No unique task directory with identifiable training data")
            subdir,names=possible[0]
            parent=parent/subdir
            locations.append((parent,names))
            splits=split_files(names,lines)
        suffix=single([Path(n).suffix.lower() for n in splits["train"]],"training format")
        spec={"metric":None,"output":"label","seed":113}
        scoped=text
        if suffix in {".jsonl",".ndjson"}:
            relative={k:single(v,f"{k} JSONL") for k,v in splits.items() if v}
            require(all(Path(v).suffix.lower() in {".jsonl",".ndjson"} for v in relative.values()),"Mixed split formats")
            raw=get_bytes(root,(parent/relative["train"]).as_posix())
            frame,kinds=jsonl_frame(io.BytesIO(raw))
            # Example payloads and bibliographies are data, not field or metric instructions.
            prose=re.sub(r"(?ms)^\s*```[^\n]*\n.*?^\s*```[^\n]*",lambda m:"\n"*m[0].count("\n"),text)
            lines=blocks(prose)
            spec.update(json_contract(frame,kinds,lines,evidence))
            target=spec["target"]
            nclasses=classification(frame[target]);shape=list(frame.shape)
            audit.update(training_columns=list(frame),json_types=kinds,excluded_columns=[c for c in frame if c not in [target,*spec["features"]]])
        elif suffix==".csv":
            relative={k:single(v,f"{k} CSV") for k,v in splits.items() if v}
            require(all(v.lower().endswith(".csv") for v in relative.values()),"Mixed split formats")
            raw=get_bytes(root,(parent/relative["train"]).as_posix())
            reader=csv.reader(io.StringIO(raw.decode("utf-8-sig"),newline=""),strict=True)
            header=next(reader,[])
            require(header and all(c.strip() for c in header) and len(set(header))==len(header),"Missing or duplicate CSV header")
            rows=list(reader)
            require(rows and all(len(row)==len(header) for row in rows),"Malformed or empty training CSV")
            frame=pd.DataFrame(rows,columns=header)
            scoped,example_target,binding=example_scope(text,frame)
            if binding:
                evidence.append(binding)
                lines=blocks(scoped)
                label_files=[n for n in names if n.casefold() in {"classes.json","categories.json"}]
                for n in label_files:
                    loc=(parent/n).as_posix()
                    values=json.loads(get_bytes(root,loc).decode("utf-8-sig"))
                    require(isinstance(values,list) and all(isinstance(v,(str,int)) for v in values)
                            and len(set(map(str,values)))==len(values) and set(frame[example_target])<=set(map(str,values)),
                            "Published label vocabulary conflicts with training labels")
                    evidence.append(dict(field="class_vocabulary",source=loc,declared=len(values),observed=frame[example_target].nunique()))
            target,fields=csv_roles(header,lines,evidence,example_target)
            nclasses=classification(frame[target])
            audit.update(training_columns=header,excluded_columns=[c for c in header if c!=target and c not in fields])
            from features import infer_modality
            pair=ordered_comparison(lines)
            if pair: require(len(fields)==2 and nclasses==2,"Pairwise comparison requires two features and two classes")
            mode="text" if pair else infer_modality(frame[fields])
            spec.update(features=fields,target=target,modality=mode)
            if pair: spec["pairwise"]=True
            if mode=="table":
                spec["numeric"]=[c for c in fields if (v:=frame[c].replace("",np.nan).dropna()).size and
                                 not v.str.match(r"^[+-]?0\d").any() and pd.to_numeric(v,errors="coerce").notna().all()]
            shape=list(frame.shape)
        else:
            relative=array_roles(splits,lines)
            raw=get_bytes(root,(parent/relative["train"]).as_posix())
            buf=io.BytesIO(raw)
            fmt=np.lib.format.read_magic(buf)
            require(fmt in ((1,0),(2,0)),"Unsupported NumPy header version")
            header=np.lib.format.read_array_header_1_0 if fmt==(1,0) else np.lib.format.read_array_header_2_0
            shape,_,dtype=header(buf)
            fields,line,excerpt=array_names(lines)
            require(len(shape)==2 and len(fields)==shape[1] and len(set(fields))==len(fields),
                    "Declared ordered fields do not match the NumPy training width")
            lraw=get_bytes(root,(parent/relative["train_labels"]).as_posix())
            labels=np.load(io.BytesIO(lraw),allow_pickle=False)
            require(labels.shape in ((shape[0],),(shape[0],1)),"NumPy training feature and label row counts differ")
            nclasses=classification(labels.reshape(-1))
            target="label"
            while target in fields: target="_"+target
            spec.update(features=fields,target=target,modality="table",numeric=fields if np.issubdtype(dtype,np.number) else [])
            if dtype.hasobject:
                spec["allow_pickle_features"]=bool(trust_objects)
                audit["object_features"]="header inspected; explicit trust required to train" if not trust_objects else "trusted by caller for training; intake inspected header only"
            audit.update(training_columns=fields,feature_dtype=str(dtype),label_dtype=str(labels.dtype))
            evidence.extend([{"field":"features","line":line,"text":excerpt},{"field":"target","source":relative["train_labels"],"value":target,"meaning":"external label array; name is internal"}])
        metric,positive_guides=resolve_metric(scoped,lines,locations,lambda n:get_bytes(root,n),cache,fetch,audit)
        spec["metric"]=metric
        if metric == "f1":
            positive = []
            for line, sentence, origin in [*((n,s,chosen) for n,s in lines),*positive_guides]:
                for m in re.finditer(r"(?:\bpositive\s+(?:class|label)|正类(?:标签)?)\s*(?:is\s+|为\s*|[:=：]\s*)[`'\x22]?([\w-]+)", sentence, re.I):
                    positive.append(m[1])
                    evidence.append({"field":"positive_class","source":origin,"line":line,"text":sentence,"value":m[1]})
            declared = single(positive, "explicit positive class in README")
            values = labels.reshape(-1) if suffix == ".npy" else frame[target]
            require(nclasses == 2 and declared in set(map(str,values)), "Positive class conflicts with training labels")
            spec["positive_class"] = declared
        spec.update({k:(parent/v).as_posix() for k,v in relative.items()})
        audit.update(readme=chosen,format=suffix,training_shape=list(shape),classes=nclasses,metric=metric)
        return spec


def generate(data, readme, out, trust_objects=False, cache=None, fetch=False, objective=None):
    report={"status":"reading","opened":[],"evidence":[]}
    report_file=out/"intake.json"
    task=out/"task.json"
    require(not task.exists() and not report_file.exists(),"Intake output already exists")
    try:
        from native_formats import inspect_native
        spec=inspect_native(data,objective,report) if readme is None else None
        if spec is None:
            spec=inspect(data,readme,trust_objects,report,cache,fetch)
            require(not objective or spec["metric"]==objective,"Requested objective conflicts with the original instructions")
        require(spec["metric"]!="f1" or spec.get("positive_class") is not None,"Binary F1 needs a declared positive class")
        report["status"]="ready" if spec["metric"] in TRAINABLE else "unsupported_training_metric"
        if spec.get("allow_pickle_features") is False:
            report["training_requires_trusted_objects"]=True
            if report["status"]=="ready":
                report["status"]="requires_trusted_objects"
        report["fields"]=spec["features"];report["target"]=spec["target"]
    except (ValueError,OSError,EOFError,csv.Error,BadZipFile) as exc:
        report.update(status="needs_clarification",reason=str(exc))
        report_file.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        raise IntakeError(str(exc)) from exc
    report_file.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    task.write_text(json.dumps(spec,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return task


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--readme",help="README path relative to the data root, required for ambiguous roots")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--trust-object-features",action="store_true")
    p.add_argument("--reference-cache",type=Path,help="Directory of fingerprinted paper PDFs")
    p.add_argument("--objective",choices=sorted(TRAINABLE),help="Assessment objective explicitly supplied with the task")
    p.add_argument("--fetch-citations",action="store_true",help="Retrieve an explicitly cited academic paper when the metric is missing")
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:
        path=generate(args.data,args.readme,args.out,args.trust_object_features,args.reference_cache,args.fetch_citations,args.objective)
        print(path)
    except IntakeError as exc:
        p.exit(2,f"{exc}\n")
