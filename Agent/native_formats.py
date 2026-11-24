"""Read published tabular schemas, classification data cards, and grayscale IDX image/label pairs."""
import gzip
import ast
import hashlib
import io
import json
from pathlib import Path
import re
import struct
from urllib.parse import urlparse

import numpy as np
import pandas as pd

def require(condition, message):
    if not condition:
        raise ValueError(message)

def only(items, role):
    items=list(items)
    require(len(items)==1, f"Expected exactly one {role}, found {len(items)}")
    return items[0]

def recorded(path, audit):
    with path.open("rb") as stream:
        digest=hashlib.file_digest(stream,"sha256").hexdigest()
    audit["opened"].append({"file":path.name,"sha256":digest,"bytes":path.stat().st_size})
    return path

def idx(path, dimensions):
    with gzip.open(path,"rb") as stream:
        header=stream.read(4+4*dimensions)
        require(len(header)==4+4*dimensions and header[:4]==bytes((0,0,8,dimensions)), "Invalid unsigned-byte IDX header or rank")
        shape=struct.unpack(">"+"I"*dimensions,header[4:])
        count=1
        for n in shape:
            count*=n
        require(all(shape) and count<=256*1024**2, "Unsupported IDX dimensions")
        raw=stream.read(count+1)
        require(len(raw)==count, "IDX payload length disagrees with its shape")
        return np.frombuffer(raw,dtype=np.uint8).reshape(shape)

def image_frame(path, field):
    pixels=idx(path,3)
    _,h,w=pixels.shape
    header=f"P5\n{w} {h}\n255\n".encode("ascii")
    return pd.DataFrame({field:[io.BytesIO(header+v.tobytes()) for v in pixels]})

def metric_candidates(values, supplied, report, location):
    values=sorted(set(values))
    report["evidence"].append(dict(field="metric",source=location,declared=values,requested=supplied))
    if supplied:
        require(not values or supplied in values, "Requested objective conflicts with published metrics")
        return supplied
    require(len(values)==1,"A single primary objective is not specified; published metrics: "+str(values))
    return values[0]

def read_parquet(path):
    try:
        from pyarrow import parquet
    except ImportError as exc:
        raise ValueError("This input requires pyarrow") from exc
    return parquet.read_table(path).to_pandas()

def evaluation_index(card):
    entries=card.get("train-eval-index",[])
    if not entries:
        return {}
    return only(entries,"evaluation configuration")

def read_card(path):
    try:
        import yaml
    except ImportError as exc:
        raise ValueError("This input requires PyYAML") from exc
    contents=path.read_text(encoding="utf-8-sig").splitlines()
    require(contents and contents[0].strip()=="---","Expected a structured dataset card")
    end=next((i for i,v in enumerate(contents[1:],1) if v.strip()=="---"),None)
    require(end is not None,"Unclosed dataset card header")
    try:
        raw=yaml.safe_load("\n".join(contents[1:end]))
    except yaml.YAMLError as exc:
        raise ValueError("Invalid YAML in dataset card") from exc
    require(isinstance(raw,dict),"Invalid card metadata")
    return raw,"\n".join(contents[end+1:])

def markdown_primary(text):
    values=[]
    for line in text.splitlines():
        m=re.fullmatch(r"\s*(?:[-*]\s*)?(?:primary|official|main)\s+(?:evaluation\s+)?(?:metric|measure)\s*(?::|is)\s*([\w -]+)\.?\s*",line.replace("**",""),re.I)
        if m:
            values.append(m[1].strip().lower().replace(" ","_"))
    return values


def scorer_metric(root, report):
    """Read the result names in a reviewed benchmark release without executing its code."""
    path = root / "benchmark" / "evaluate.py"
    if not path.is_file():
        return []
    recorded(path, report)
    # DialoGLUE revision 9074e4f27c973c087e9b814ce286219c6643380b.
    # The published rename table drives its output; Acc is per-row exact label agreement.
    if report["opened"][-1]["sha256"] != "72bd352fe80d84edcb23d6d8dfff59efc366b75d4337660e79d4fe5acc85016c":
        return []
    module = ast.parse(path.read_bytes())
    evaluator = only((fn for fn in module.body if isinstance(fn, ast.FunctionDef) and fn.name == "evaluate"), "evaluation function")
    table = only((node.value for node in evaluator.body if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "rename" for t in node.targets)), "result name mapping")
    def normalize(name):
        return re.sub(r"[^a-z0-9]", "", name.casefold())
    identity = normalize(root.name)
    matches = []
    for key, label in ast.literal_eval(table).items():
        match = re.fullmatch(r"(.+?)\s*\(([^()]+)\)", label)
        if match and identity in {normalize(key), normalize(match[1])}:
            matches.append((key, match[2]))
    if len(matches) != 1 or matches[0][1] != "Acc":
        return []
    report["evidence"].append(dict(field="scorer_binding", source="benchmark/evaluate.py", directory=root.name,
                                   benchmark_key=matches[0][0], result_name=ast.literal_eval(table)[matches[0][0]]))
    return ["accuracy"]

def card_spec(root, train, supplied, report):
    readme=recorded(only((p for p in root.iterdir() if p.name.lower()=="readme.md"),"dataset card"),report)
    card,body=read_card(readme)
    info=card.get("dataset_info",{})
    require(isinstance(info,dict),"Multiple dataset configurations require an explicit selection")
    fields=info.get("features",[])
    require(isinstance(fields,list) and all(isinstance(f,dict) and "name" in f and "dtype" in f for f in fields), "Unsupported card schema")
    evaluation=evaluation_index(card)
    roles=evaluation.get("col_mapping",{})
    enums=[f["name"] for f in fields if isinstance(f["dtype"],dict) and "class_label" in f["dtype"]]
    frame=read_parquet(recorded(train,report))
    names=[f["name"] for f in fields]
    require(list(frame)==names and len(set(names))==len(names) and isinstance(roles,dict)
            and set(roles)<=set(names),"Card columns disagree with training schema")
    targets=[k for k,v in roles.items() if v=="target"] or enums
    role_source="dataset_info.features / train-eval-index.col_mapping"
    if not targets:
        targets=[k for k in names if k.casefold() in {"label","labels","class","target"} and k not in roles]
        role_source="declared column name and observed categorical training values"
        target=only(targets,"classification target")
        labels=frame[target]
        dtype=next(f["dtype"] for f in fields if f["name"]==target)
        require(isinstance(dtype,str) and (dtype in {"string","bool"} or re.fullmatch(r"u?int(?:8|16|32|64)",dtype))
                and not labels.isna().any() and labels.astype(str).str.strip().ne("").all()
                and 1<labels.nunique()<=len(labels)//2,"Named target does not have recurring categorical training labels")
    else:
        target=only(targets,"classification target")
    require((not enums or enums==[target]) and roles.get(target,"target")=="target","Conflicting or multiple label fields")
    declared_inputs=[k for k in names if k!=target and roles.get(k) not in {None,"id","index","target"}]
    features=declared_inputs or [k for k in names if k!=target and roles.get(k) not in {"id","index","target"}
                                 and k.casefold() not in {"id","idx","index"}]
    require(features and not any(k in enums for k in features),"Ambiguous input mapping")
    require(all(isinstance(f["dtype"],str) for f in fields if f["name"] in features),"Unsupported nested input column")
    from features import infer_modality
    mode=infer_modality(frame[features].fillna("").astype(str))
    metric_set=[]
    for m in evaluation.get("metrics",[]):
        method=m["type"];avg=m.get("args",{}).get("average")
        metric_set.append(f"{avg}_{method}" if avg and avg!="binary" else method)
    primary=markdown_primary(body)
    require(not primary or not metric_set or set(primary)<=set(metric_set),"Card disagrees about evaluation metrics")
    report["evidence"].append({"field":"roles","source":role_source,"target":target,"features":features,"training_classes":int(frame[target].nunique())})
    benchmark=scorer_metric(root,report)
    declared=primary or metric_set
    require(not declared or not benchmark or set(benchmark)<=set(declared),"Card and published scorer disagree")
    metric=metric_candidates(primary or benchmark or metric_set,supplied,report,readme.name if primary or not benchmark else "benchmark/evaluate.py")
    require(not frame[target].isna().any() and frame[target].nunique()>1,"Invalid training labels")
    report.update(training_shape=list(frame.shape),classes=int(frame[target].nunique()))
    spec=dict(target=target,features=features,modality=mode,metric=metric,train=train.relative_to(root).as_posix())
    if mode=="table":spec["numeric"]=[col for col in features if pd.api.types.is_numeric_dtype(frame[col])]
    for name in ("validation","valid","val","dev"):
        candidates=list(train.parent.glob(name+".parquet"))+list(train.parent.glob(name+"-*.parquet"))
        if candidates:
            require("validation" not in spec,"Multiple validation splits")
            spec["validation"]=only(candidates,"validation shard").relative_to(root).as_posix()
    return spec

def table_spec(root, meta, supplied, report):
    obj=json.loads(recorded(meta,report).read_text(encoding="utf-8-sig"))["data"]
    require(isinstance(obj,dict) and "Classification" in obj.get("tasks",[]),"Metadata does not declare a classification task")
    data_name=Path(urlparse(obj["data_url"]).path).name
    path=root/data_name
    require(path.suffix.lower()==".csv" and path.is_file(),"Missing publisher-declared CSV")
    variables=obj["variables"]
    names=[v["name"] for v in variables]
    frame=pd.read_csv(recorded(path,report),dtype=str,keep_default_na=False)
    require(len(set(names))==len(names) and set(names)==set(frame),"CSV header disagrees with metadata variables")
    target=only((v["name"] for v in variables if v["role"]=="Target"),"target field")
    require(obj.get("target_col")==[target],"Target declarations disagree")
    features=[v["name"] for v in variables if v["role"]=="Feature"]
    require(features and frame[target].nunique()>1 and frame[target].str.strip().ne("").all(),"Invalid features or labels")
    numeric=[v["name"] for v in variables if v["role"]=="Feature" and v["type"].lower() in {"continuous","integer","real","numeric"}]
    report.update(training_shape=list(frame.shape),classes=int(frame[target].nunique()))
    report["evidence"].append({"field":"roles","source":meta.name+" variables.role and target_col","target":target,"features":features})
    primary=markdown_primary("\n".join(str(v) for v in (obj.get("additional_info") or {}).values() if v))
    return dict(train=data_name,features=features,numeric=numeric,target=target,modality="table",
                metric=metric_candidates(primary,supplied,report,meta.name))

def image_spec(root,files,supplied,report):
    doc=recorded(only((p for p in root.iterdir() if p.name.lower()=="readme.md"),"IDX task README"),report)
    lines=doc.read_text(encoding="utf-8-sig").splitlines()
    image=only((p for p in files if "idx3" in p.name.lower()),"rank-three training image array")
    label=only((p for p in files if "idx1" in p.name.lower()),"rank-one training label array")
    for file,pattern in ((image,r"\bimages?\b"),(label,r"\blabels?\b")):
        require(any(file.name in line and re.search(pattern,line.replace(file.name,""),re.I) for line in lines),"README does not bind IDX file to its semantic role")
    images=idx(recorded(image,report),3); labels=idx(recorded(label,report),1)
    require(len(labels)==len(images) and len(set(labels))>1,"IDX rows or labels disagree")
    measures=[]
    for i,line in enumerate(lines[:-1]):
        if line.startswith("|") and re.fullmatch(r"[\s:|\-]+",lines[i+1]) and re.search(r"\b(classifier|model|algorithm)\b",line,re.I):
            for cell in line.lower().split("|"):
                for term in ("balanced accuracy","accuracy","macro f1","f1","precision","recall","auc","error rate"):
                    if re.search(r"\b"+term+r"\b",cell):
                        measures.append(term.replace(" ","_"));break
            if measures:report["evidence"].append({"field":"benchmark_columns","line":i+1,"text":line})
    declared=markdown_primary("\n".join(lines))
    require(not declared or not measures or set(declared)<=set(measures),"IDX benchmark metrics disagree with primary measure")
    report.update(training_shape=list(images.shape),classes=len(set(labels)))
    spec=dict(train=image.name,train_labels=label.name,target="label",features=["image"],modality="image",native_format="idx",
              metric=metric_candidates(declared or measures,supplied,report,doc.name))
    val=[p for p in root.glob("*.gz") if re.match(r"^(val|valid|validation|dev)[_-]",p.name,re.I)]
    if val:
        spec.update(validation=only((p for p in val if "idx3" in p.name),"validation image array").name,
                    validation_labels=only((p for p in val if "idx1" in p.name),"validation label array").name)
    return spec

def inspect_native(root,supplied,report):
    if not root.is_dir():return None
    meta=[p for p in root.iterdir() if p.name.lower() in {"metadata.json","dataset.json"}]
    parquet=[p for p in root.glob("**/*.parquet") if p.is_file() and re.fullmatch(r"train(?:ing)?(?:-\d+-of-\d+)?\.parquet",p.name,re.I)
             and not any(v.lower() in {"test","validation","valid","dev"} for v in p.relative_to(root).parts[:-1])]
    images=[p for p in root.glob("*.gz") if re.match(r"^train(?:ing)?[_-].*idx[13][_-]ubyte\.gz$",p.name,re.I)]
    require(sum(map(bool,(meta,parquet,images)))<=1,"Multiple incompatible native training schemas")
    if parquet:spec=card_spec(root,only(parquet,"training shard"),supplied,report)
    elif images:spec=image_spec(root,images,supplied,report)
    elif meta:
        path=only(meta,"publisher metadata file")
        candidate=json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(candidate,dict) or not isinstance(candidate.get("data"),dict) or "variables" not in candidate["data"]:return None
        spec=table_spec(root,path,supplied,report)
    else:return None
    report.update(metric=spec["metric"],format=Path(spec["train"]).suffix)
    return {**spec,"output":"label","seed":113}
