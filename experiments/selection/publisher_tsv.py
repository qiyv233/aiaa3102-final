"""Read a local publisher's registered single-text classification task without importing its code."""
import ast
import io
from pathlib import Path, PurePosixPath
import re

from task_data import data_file, headerless_frame
from native_formats import require


def one(items, description):
    matches = list(items)
    require(len(matches) == 1, f"Expected one {description}, found {len(matches)}")
    return matches[0]


def name(node):
    return node.id if isinstance(node, ast.Name) else None


def constant(node):
    return node.value if isinstance(node, ast.Constant) else None


def assignments(body):
    return [(stmt.targets[0], stmt.value) for stmt in body
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1]


def fields(body, owner):
    result = {}
    for target, value in assignments(body):
        if isinstance(target, ast.Subscript) and name(target.value) == owner:
            key = constant(target.slice)
            require(isinstance(key, str) and key not in result, "Ambiguous publisher configuration key")
            result[key] = value
    return result


def imports(tree):
    bindings = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                local = alias.asname or alias.name
                require(local not in bindings, f"Ambiguous import for {local}")
                bindings[local] = (node.module, alias.name)
    return bindings


def symbol(tree, kind, symbol_name):
    return one((node for node in tree.body if isinstance(node, kind) and node.name == symbol_name),
               f"definition of {symbol_name}")


def literal(node, message):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        raise ValueError(message) from None


def local_path(root, parent, relative, *, allow_validation=False):
    require(isinstance(relative, str) and relative and not re.search(r"[:\\\x00]", relative),
            "Publisher paths must be local relative paths")
    path = PurePosixPath(relative)
    require(not path.is_absolute() and all(not p.startswith(".") for p in path.parts),
            "Publisher path outside the task directory")
    tokens = set(re.findall(r"[a-z0-9]+", "/".join(path.parts).lower()))
    heldout = {"test","testing","t10k","heldout"} if allow_validation else {"test","testing","t10k","val","valid","validation","dev","heldout"}
    require(not tokens.intersection(heldout), "Publisher path points to a different split")
    relative_path = (parent/path).as_posix()
    candidate = data_file(root, relative_path)
    if isinstance(candidate, Path):
        require(candidate.resolve() == root.resolve()/relative_path,
                "Publisher path is redirected outside the declared layout")
    return relative_path, candidate


def registration(root, parent, names, parse):
    entries = [key for key in ("main.py", "train.py") if key in names]
    matches = []
    for entry in entries:
        entry_path, _ = local_path(root, parent, entry)
        tree = parse(entry_path)
        imported = imports(tree)
        # Recognize the registration entry point by data flow: cfg = configure(cfg).
        calls = {name(node.value.func) for node in ast.walk(tree) if isinstance(node, ast.Assign)
                 and len(node.targets) == 1 and name(node.targets[0]) and isinstance(node.value, ast.Call)
                 and len(node.value.args) == 1 and name(node.value.args[0]) == name(node.targets[0])}
        for local in sorted(calls.intersection(imported)):
            module, function = imported[local]
            file, path = local_path(root, parent, module.replace(".", "/")+".py")
            if not path.is_file():
                continue
            config = parse(file)
            fn = symbol(config, ast.FunctionDef, function)
            if not fn.args.args:
                continue
            owner = fn.args.args[0].arg
            for node in ast.walk(fn):
                if not isinstance(node, ast.If):
                    continue
                setting = fields(node.body, owner)
                if "train_set_path" not in setting:
                    continue
                raw = constant(setting["train_set_path"])
                require(isinstance(raw, str), "Dynamic publisher training path")
                train, physical = local_path(root, parent, raw)
                if physical.is_file():
                    matches.append((file, setting, train))
    if not matches:
        return None
    # Two entry points may call the same registry; two blocks may not claim one file.
    unique = {(file, rule["train_set_path"].lineno, train):(file, rule, train) for file,rule,train in matches}
    return one(unique.values(), "available registered training task")


def loader_schema(tree, cls):
    method = symbol(cls, ast.FunctionDef, "load_dataset")
    body = [node for node in method.body if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                                                  and isinstance(node.value.value,str))]
    require(len(body) == 4 and len(assignments(body)) == 3 and isinstance(body[-1], ast.Return),
            "Unsupported transformations in the publisher's dataset loader")
    (frame, read), (columns, schema), (column, transform) = assignments(body)
    df = name(frame)
    pandas_names = {alias.asname or "pandas" for node in tree.body if isinstance(node, ast.Import)
                    for alias in node.names if alias.name == "pandas"}
    require(df and isinstance(read, ast.Call) and isinstance(read.func, ast.Attribute)
            and name(read.func.value) in pandas_names and read.func.attr == "read_csv"
            and len(method.args.args) == 2 and len(read.args) == 1 and name(read.args[0]) == method.args.args[1].arg
            and len(read.keywords)==2 and {k.arg:literal(k.value,"Dynamic TSV reader option") for k in read.keywords}
            == {"header":None,"sep":"\t"}, "Loader must declare a headerless tab-separated file")
    require(isinstance(columns, ast.Attribute) and name(columns.value) == df and columns.attr == "columns"
            and name(body[-1].value) == df, "TSV schema is not assigned to the returned frame")
    order = literal(schema, "Dynamic TSV field names")
    require(isinstance(order,(list,tuple)) and len(order)==2 and all(isinstance(c,str) and c.strip() for c in order)
            and len(set(order))==2, "Expected one input and one target field")
    require(isinstance(column, ast.Subscript) and name(column.value)==df and constant(column.slice) in order,
            "Unbound label field")
    target = constant(column.slice)
    require(isinstance(transform,ast.Call) and isinstance(transform.func,ast.Attribute) and transform.func.attr=="apply"
            and ast.unparse(transform.func.value)==ast.unparse(column) and len(transform.args)==1 and not transform.keywords,
            "Unrecognized target mapping")
    func=transform.args[0]
    require(isinstance(func,ast.Lambda) and len(func.args.args)==1 and isinstance(func.body,ast.Subscript)
            and name(func.body.slice)==func.args.args[0].arg and isinstance(func.body.value,ast.Attribute)
            and name(func.body.value.value)==method.args.args[0].arg, "Target mapping is not a label lookup")
    map_name=func.body.value.attr
    mapping=literal(one((v for k,v in assignments(cls.body) if name(k)==map_name),"label vocabulary"),
                    "Dynamic label vocabulary")
    require(isinstance(mapping,dict) and len(mapping)>1 and all(isinstance(k,str) for k in mapping)
            and all(type(v) is int for v in mapping.values()) and len(set(mapping.values()))==len(mapping),
            "Invalid class vocabulary")
    return list(order),target,mapping,schema.lineno


def scoring_rule(tree, fn, criterion):
    require(isinstance(fn.body[-1],ast.Return) and name(fn.body[-1].value)
            and all(isinstance(node,(ast.Assign,ast.Return,ast.Expr)) for node in fn.body),
            "Unsupported conditional scoring function")
    output = name(fn.body[-1].value)
    initializers = [v for k,v in assignments(fn.body) if name(k)==output]
    require(len(initializers)==1 and isinstance(initializers[0],ast.Dict) and not initializers[0].keys
            and isinstance(fn.body[0],ast.Assign) and fn.body[0].value is initializers[0]
            and all(isinstance(s,ast.Assign) and len(s.targets)==1 and isinstance(s.targets[0],ast.Subscript)
                    and name(s.targets[0].value)==output for s in fn.body[1:-1]),
            "Scoring dictionary must return the declared measures directly")
    returned = fields(fn.body,output)
    call = returned.get(criterion)
    require(isinstance(call,ast.Call) and name(call.func) and len(call.args)==2 and
            len(fn.args.args)==2 and {name(v) for v in call.args}=={v.arg for v in fn.args.args},
            "Validation criterion must score both original arguments")
    source = imports(tree).get(name(call.func))
    options = {k.arg:literal(k.value,"Dynamic scoring option") for k in call.keywords}
    if source == ("sklearn.metrics","accuracy_score") and not options:
        return "accuracy",call.lineno
    require(source==("sklearn.metrics","f1_score") and set(options)=={"average"}
            and options["average"] in ("macro","weighted","micro"), "Unresolved scoring function or averaging rule")
    return options["average"]+"_f1",call.lineno


def registered_task(root, parent, names, get_bytes, audit):
    trees = {}
    def parse(path):
        if path not in trees:
            try:
                trees[path] = ast.parse(get_bytes(root,path).decode("utf-8-sig"))
            except SyntaxError as exc:
                raise ValueError(f"Invalid publisher source in {path}") from exc
        return trees[path]
    found = registration(root,parent,names,parse)
    if found is None:
        return None
    config, rule, train = found
    require(train.lower().endswith(".tsv") and constant(rule.get("task"))=="sequence_classification",
            "Registered loader supports single-text classification TSVs")
    bindings = imports(parse(config))
    def definition(key, kind):
        binding = bindings.get(name(rule.get(key)))
        require(binding, f"Missing local definition for {key}")
        module,identifier = binding
        path,_ = local_path(root,parent,module.replace(".","/")+".py")
        tree = parse(path)
        return path,tree,symbol(tree,kind,identifier)
    loader, tree, cls = definition("dataset_class",ast.ClassDef)
    order,target,labels,line = loader_schema(tree,cls)
    scorer,tree,fn = definition("metrics_fn",ast.FunctionDef)
    criterion=constant(rule.get("valid_criterion"))
    metric,metric_line=scoring_rule(tree,fn,criterion)
    features=[c for c in order if c!=target]
    spec=dict(train=train,tsv_columns=order,target=target,features=features,metric=metric,
              native_format="headerless_tsv",modality="text",output="label",seed=113)
    if "valid_set_path" in rule:
        vpath, val = local_path(root,parent,constant(rule["valid_set_path"]),allow_validation=True)
        require(vpath!=train, "Training and validation paths coincide")
        if val.is_file():
            spec["validation"]=vpath
    frame=headerless_frame(io.BytesIO(get_bytes(root,train)),spec,labeled=True)
    require(frame[target].nunique()>1 and set(frame[target])<=set(labels),"Training labels disagree with the publisher vocabulary")
    audit["evidence"].extend([
        dict(field="train",source=config,line=rule["train_set_path"].lineno,value=train),
        dict(field="roles",source=loader,line=line,columns=order,target=target,features=features),
        dict(field="criterion",source=config,line=rule["valid_criterion"].lineno,value=criterion),
        dict(field="metric",source=scorer,line=metric_line,value=metric)])
    audit.update(format="headerless_tsv",training_shape=list(frame.shape),classes=frame[target].nunique(),
                 training_columns=order,metric=metric)
    return spec
