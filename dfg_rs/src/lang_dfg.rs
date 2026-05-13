use std::collections::HashMap;
use tree_sitter::Node;
use crate::extractor::{DfgNode, TokenIndex, find_token_idx};

pub struct LangConfig {
    pub def_stmts:     &'static [&'static str],
    pub assignments:   &'static [&'static str],
    pub increments:    &'static [&'static str],
    pub if_stmts:      &'static [&'static str],
    pub for_stmts:     &'static [&'static str],
    pub for_trigger:   &'static str,
    pub enhanced_fors: &'static [&'static str],
    pub while_stmts:   &'static [&'static str],
    pub ef_name:       &'static str,
    pub ef_value:      &'static str,
    pub ef_body:       &'static str,
    pub def_val_field: &'static str,
    pub csharp_def:    bool,
    pub php_foreach:   bool,
    pub ruby_op:       bool,
    pub go_for:        bool,
}

pub static JAVA: LangConfig = LangConfig {
    def_stmts: &["variable_declarator"],
    assignments: &["assignment_expression"],
    increments: &["update_expression"],
    if_stmts: &["if_statement", "else"],
    for_stmts: &["for_statement"],
    for_trigger: "local_variable_declaration",
    enhanced_fors: &["enhanced_for_statement"],
    while_stmts: &["while_statement"],
    ef_name: "name", ef_value: "value", ef_body: "body",
    def_val_field: "value",
    csharp_def: false, php_foreach: false, ruby_op: false, go_for: false,
};

pub static CSHARP: LangConfig = LangConfig {
    def_stmts: &["variable_declarator"],
    assignments: &["assignment_expression"],
    increments: &["postfix_unary_expression"],
    if_stmts: &["if_statement", "else"],
    for_stmts: &["for_statement"],
    for_trigger: "local_variable_declaration",
    enhanced_fors: &["for_each_statement"],
    while_stmts: &["while_statement"],
    ef_name: "left", ef_value: "right", ef_body: "body",
    def_val_field: "value",
    csharp_def: true, php_foreach: false, ruby_op: false, go_for: false,
};

pub static RUBY: LangConfig = LangConfig {
    def_stmts: &["keyword_parameter"],
    assignments: &["assignment", "operator_assignment"],
    increments: &[],
    if_stmts: &["if", "elsif", "else", "unless", "when"],
    for_stmts: &["for"],
    for_trigger: "",
    enhanced_fors: &[],
    while_stmts: &["while_modifier", "until"],
    ef_name: "pattern", ef_value: "value", ef_body: "body",
    def_val_field: "value",
    csharp_def: false, php_foreach: false, ruby_op: true, go_for: false,
};

pub static GO: LangConfig = LangConfig {
    def_stmts: &["var_spec"],
    assignments: &["assignment_statement"],
    increments: &["inc_statement"],
    if_stmts: &["if_statement", "else"],
    for_stmts: &["for_statement"],
    for_trigger: "for_clause",
    enhanced_fors: &[],
    while_stmts: &[],
    ef_name: "name", ef_value: "value", ef_body: "body",
    def_val_field: "value",
    csharp_def: false, php_foreach: false, ruby_op: false, go_for: true,
};

pub static PHP: LangConfig = LangConfig {
    def_stmts: &["simple_parameter"],
    assignments: &["assignment_expression", "augmented_assignment_expression"],
    increments: &["update_expression"],
    if_stmts: &["if_statement", "else_clause"],
    for_stmts: &["for_statement"],
    for_trigger: "assignment_expression",
    enhanced_fors: &["foreach_statement"],
    while_stmts: &["while_statement"],
    ef_name: "name", ef_value: "value", ef_body: "body",
    def_val_field: "default_value",
    csharp_def: false, php_foreach: true, ruby_op: false, go_for: false,
};

pub static JS: LangConfig = LangConfig {
    def_stmts: &["variable_declarator"],
    assignments: &["assignment_pattern", "augmented_assignment_expression"],
    increments: &["update_expression"],
    if_stmts: &["if_statement", "else"],
    for_stmts: &["for_statement"],
    for_trigger: "variable_declaration",
    enhanced_fors: &[],
    while_stmts: &["while_statement"],
    ef_name: "name", ef_value: "value", ef_body: "body",
    def_val_field: "value",
    csharp_def: false, php_foreach: false, ruby_op: false, go_for: false,
};

fn has(list: &[&str], s: &str) -> bool { list.contains(&s) }

pub fn var_indices(node: Node, tokens: &[TokenIndex]) -> Vec<usize> {
    let mut r = Vec::new();
    var_indices_inner(node, tokens, &mut r);
    r
}

fn var_indices_inner(node: Node, tokens: &[TokenIndex], r: &mut Vec<usize>) {
    let is_leaf = node.child_count() == 0;
    let is_str  = node.kind() == "string";
    let is_cmt  = node.kind() == "comment";
    if (is_leaf || is_str) && !is_cmt {
        let p = node.start_position();
        let e = node.end_position();
        if let Some(idx) = find_token_idx(tokens, p.row, p.column, e.row, e.column) {
            if node.kind() != tokens[idx].token.as_str() { r.push(idx); }
        }
        return;
    }
    let mut cur = node.walk();
    for child in node.children(&mut cur) { var_indices_inner(child, tokens, r); }
}

pub fn dedup(mut dfg: Vec<DfgNode>) -> Vec<DfgNode> {
    dfg.sort_by_key(|n| n.index);
    let mut seen: HashMap<(String, usize), usize> = HashMap::new();
    let mut out: Vec<DfgNode> = Vec::new();
    for node in dfg {
        let key = (node.name.clone(), node.index);
        if let Some(&pos) = seen.get(&key) {
            for s in &node.sources {
                if !out[pos].sources.contains(s) { out[pos].sources.push(*s); }
            }
            out[pos].sources.sort();
        } else {
            seen.insert(key, out.len());
            out.push(node);
        }
    }
    out
}

pub fn extract(node: Node, tokens: &[TokenIndex], states: &mut HashMap<String, Vec<usize>>,
               result: &mut Vec<DfgNode>, cfg: &'static LangConfig) {
    let is_leaf = node.child_count() == 0;
    let is_str  = node.kind() == "string";
    let is_cmt  = node.kind() == "comment";

    if (is_leaf || is_str) && !is_cmt {
        let p = node.start_position();
        let e = node.end_position();
        if let Some(idx) = find_token_idx(tokens, p.row, p.column, e.row, e.column) {
            let tok  = tokens[idx].token.clone();
            let kind = node.kind();
            if kind == tok.as_str() { return; }
            if states.contains_key(&tok) {
                let srcs = states[&tok].clone();
                result.push(DfgNode { name: tok, index: idx, sources: srcs });
            } else {
                if kind == "identifier" { states.insert(tok.clone(), vec![idx]); }
                result.push(DfgNode { name: tok, index: idx, sources: vec![] });
            }
        }
        return;
    }

    let kind = node.kind();
    if has(cfg.def_stmts,     kind) { def_stmt(node, tokens, states, result, cfg); return; }
    if has(cfg.assignments,   kind) { assignment(node, tokens, states, result, cfg); return; }
    if has(cfg.increments,    kind) { increment(node, tokens, states, result); return; }
    if has(cfg.if_stmts,      kind) { if_stmt(node, tokens, states, result, cfg); return; }
    if has(cfg.for_stmts,     kind) { for_stmt(node, tokens, states, result, cfg); return; }
    if has(cfg.enhanced_fors, kind) { enhanced_for(node, tokens, states, result, cfg); return; }
    if has(cfg.while_stmts,   kind) { while_stmt(node, tokens, states, result, cfg); return; }

    let mut cur = node.walk();
    for child in node.children(&mut cur) {
        extract(child, tokens, states, result, cfg);
    }
}

fn def_stmt(node: Node, tokens: &[TokenIndex], states: &mut HashMap<String, Vec<usize>>,
            result: &mut Vec<DfgNode>, cfg: &'static LangConfig) {
    let (name_node, val_node) = if cfg.csharp_def {
        (node.child(0), if node.child_count() >= 2 { node.child(1) } else { None })
    } else {
        (node.child_by_field_name("name"), node.child_by_field_name(cfg.def_val_field))
    };
    let name_node = match name_node { Some(n) => n, None => return };
    if val_node.is_none() {
        for idx in var_indices(name_node, tokens) {
            let tok = tokens[idx].token.clone();
            result.push(DfgNode { name: tok.clone(), index: idx, sources: vec![] });
            states.insert(tok, vec![idx]);
        }
        return;
    }
    let val_node = val_node.unwrap();
    let n_idxs = var_indices(name_node, tokens);
    let v_idxs = var_indices(val_node, tokens);
    extract(val_node, tokens, states, result, cfg);
    for &ni in &n_idxs {
        let tok = tokens[ni].token.clone();
        // Python: for index2 in value_indexs → 하나당 entry 하나 생성
        if v_idxs.is_empty() {
            result.push(DfgNode { name: tok.clone(), index: ni, sources: vec![] });
        } else {
            for &vi in &v_idxs {
                result.push(DfgNode { name: tok.clone(), index: ni, sources: vec![vi] });
            }
        }
        states.insert(tok, vec![ni]);
    }
}

fn assignment(node: Node, tokens: &[TokenIndex], states: &mut HashMap<String, Vec<usize>>,
              result: &mut Vec<DfgNode>, cfg: &'static LangConfig) {
    let (left, right) = if cfg.ruby_op && node.kind() == "operator_assignment" {
        (node.child(0), node.child(node.child_count().saturating_sub(1)))
    } else {
        (node.child_by_field_name("left"), node.child_by_field_name("right"))
    };
    let left  = match left  { Some(n) => n, None => return };
    let right = match right { Some(n) => n, None => return };
    extract(right, tokens, states, result, cfg);
    let l_idxs = var_indices(left, tokens);
    let r_idxs = var_indices(right, tokens);
    for &li in &l_idxs {
        let tok = tokens[li].token.clone();
        result.push(DfgNode { name: tok.clone(), index: li, sources: r_idxs.clone() });
        states.insert(tok, vec![li]);
    }
}

fn increment(node: Node, tokens: &[TokenIndex], states: &mut HashMap<String, Vec<usize>>,
             result: &mut Vec<DfgNode>) {
    let idxs = var_indices(node, tokens);
    for &i in &idxs {
        let tok = tokens[i].token.clone();
        result.push(DfgNode { name: tok.clone(), index: i, sources: idxs.clone() });
        states.insert(tok, vec![i]);
    }
}

fn if_stmt(node: Node, tokens: &[TokenIndex], states: &mut HashMap<String, Vec<usize>>,
           result: &mut Vec<DfgNode>, cfg: &'static LangConfig) {
    let mut cur_st = states.clone();
    let mut others: Vec<HashMap<String, Vec<usize>>> = Vec::new();
    let mut tag = node.kind().contains("else");
    let mut flag = false;
    let mut curs = node.walk();
    for child in node.children(&mut curs) {
        if child.kind().contains("else") { tag = true; }
        let is_branch = has(cfg.if_stmts, child.kind());
        let is_elif   = child.kind() == "elif_clause" || child.kind() == "else_clause"
                     || (is_branch && flag);
        if is_elif {
            let mut ns = states.clone();
            extract(child, tokens, &mut ns, result, cfg);
            others.push(ns);
        } else {
            extract(child, tokens, &mut cur_st, result, cfg);
            if is_branch { flag = true; }
        }
    }
    others.push(cur_st);
    if !tag { others.push(states.clone()); }
    let mut new_st: HashMap<String, Vec<usize>> = HashMap::new();
    for dic in &others {
        for (k, v) in dic { new_st.entry(k.clone()).or_default().extend(v); }
    }
    if cfg.go_for {
        for (k, v) in states.iter() { new_st.entry(k.clone()).or_default().extend(v); }
    }
    for v in new_st.values_mut() { v.sort(); v.dedup(); }
    *states = new_st;
}

fn for_stmt(node: Node, tokens: &[TokenIndex], states: &mut HashMap<String, Vec<usize>>,
            result: &mut Vec<DfgNode>, cfg: &'static LangConfig) {
    let mut dfg: Vec<DfgNode> = Vec::new();
    let mut curs = node.walk();
    for child in node.children(&mut curs) { extract(child, tokens, states, &mut dfg, cfg); }
    if cfg.go_for {
        let mut curs = node.walk();
        for child in node.children(&mut curs) {
            if child.kind() == "for_clause" {
                if let Some(upd) = child.child_by_field_name("update") {
                    extract(upd, tokens, states, &mut dfg, cfg);
                }
            }
        }
    } else if !cfg.for_trigger.is_empty() {
        let mut flag = false;
        let mut curs = node.walk();
        for child in node.children(&mut curs) {
            if flag { extract(child, tokens, states, &mut dfg, cfg); }
            else if child.kind() == cfg.for_trigger { flag = true; }
        }
    }
    result.extend(dedup(dfg));
}

fn enhanced_for(node: Node, tokens: &[TokenIndex], states: &mut HashMap<String, Vec<usize>>,
                result: &mut Vec<DfgNode>, cfg: &'static LangConfig) {
    let mut dfg: Vec<DfgNode> = Vec::new();
    if cfg.php_foreach {
        let vars: Vec<Node> = { let mut c = node.walk();
            node.children(&mut c).filter(|ch| ch.kind() == "variable_name").collect() };
        let (val, nam) = match (vars.get(0), vars.get(1)) { (Some(&a), Some(&b)) => (a,b), _ => return };
        let body = node.child_by_field_name("body");
        for _ in 0..2 {
            extract(val, tokens, states, &mut dfg, cfg);
            let n_idxs = var_indices(nam, tokens);
            let v_idxs = var_indices(val, tokens);
            for &ni in &n_idxs {
                let tok = tokens[ni].token.clone();
                dfg.push(DfgNode { name: tok.clone(), index: ni, sources: v_idxs.clone() });
                states.insert(tok, vec![ni]);
            }
            if let Some(b) = body { extract(b, tokens, states, &mut dfg, cfg); }
        }
    } else {
        let nam  = node.child_by_field_name(cfg.ef_name);
        let val  = node.child_by_field_name(cfg.ef_value);
        let body = node.child_by_field_name(cfg.ef_body);
        let (nam, val) = match (nam, val) { (Some(a), Some(b)) => (a,b), _ => return };
        for _ in 0..2 {
            extract(val, tokens, states, &mut dfg, cfg);
            let n_idxs = var_indices(nam, tokens);
            let v_idxs = var_indices(val, tokens);
            for &ni in &n_idxs {
                let tok = tokens[ni].token.clone();
                dfg.push(DfgNode { name: tok.clone(), index: ni, sources: v_idxs.clone() });
                states.insert(tok, vec![ni]);
            }
            if let Some(b) = body { extract(b, tokens, states, &mut dfg, cfg); }
        }
    }
    result.extend(dedup(dfg));
}

fn while_stmt(node: Node, tokens: &[TokenIndex], states: &mut HashMap<String, Vec<usize>>,
              result: &mut Vec<DfgNode>, cfg: &'static LangConfig) {
    let mut dfg: Vec<DfgNode> = Vec::new();
    for _ in 0..2 {
        let mut curs = node.walk();
        for child in node.children(&mut curs) { extract(child, tokens, states, &mut dfg, cfg); }
    }
    result.extend(dedup(dfg));
}
