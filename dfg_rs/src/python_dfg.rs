// python_dfg.rs — DFG_python 정확한 재구현
// Python 원본 DFG.py의 DFG_python 로직을 Rust로 구현

use std::collections::HashMap;
use tree_sitter::Node;
use crate::extractor::{DfgNode, TokenIndex, find_token_idx};

pub fn extract_dfg_python(
    node:    Node,
    tokens:  &[TokenIndex],
    states:  &mut HashMap<String, Vec<usize>>,
    result:  &mut Vec<DfgNode>,
) {
    let is_leaf    = node.child_count() == 0;
    let is_string  = node.kind() == "string";
    let is_comment = node.kind() == "comment";

    // ── 리프 케이스 (Python DFG_python과 동일) ─────────────────────
    // Python: if (leaf or string) and not comment
    if (is_leaf || is_string) && !is_comment {
        let pos  = node.start_position();
        let epos = node.end_position();
        if let Some(idx) = find_token_idx(tokens, pos.row, pos.column, epos.row, epos.column) {
            let token_text = tokens[idx].token.clone();
            let node_kind  = node.kind();

            // Python: if root_node.type == code → skip (키워드/연산자)
            // 예: type='def', code='def' → skip
            //     type='identifier', code='x' → include
            //     type='integer', code='1' → include
            if node_kind == token_text {
                return;
            }

            // Python: elif code in states → comesFrom with sources
            if states.contains_key(&token_text) {
                let sources = states[&token_text].clone();
                result.push(DfgNode { name: token_text, index: idx, sources });
            } else {
                // Python: else → comesFrom with no sources
                // identifier이면 states 업데이트
                if node_kind == "identifier" {
                    states.insert(token_text.clone(), vec![idx]);
                }
                result.push(DfgNode { name: token_text, index: idx, sources: vec![] });
            }
        }
        return;
    }

    // ── 특수 케이스: assignment (sources 관계 설정) ─────────────────
    match node.kind() {
        "assignment" | "augmented_assignment" => {
            handle_assignment(node, tokens, states, result);
        }

        "for_statement" => {
            handle_for_statement(node, tokens, states, result);
        }

        _ => {
            // Default: 자식 재귀 (Python의 나머지 케이스)
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                extract_dfg_python(child, tokens, states, result);
            }
        }
    }
}

// ── assignment 처리 ─────────────────────────────────────────────────
fn handle_assignment(
    node:    Node,
    tokens:  &[TokenIndex],
    states:  &mut HashMap<String, Vec<usize>>,
    result:  &mut Vec<DfgNode>,
) {
    // 오른쪽 먼저 처리
    if let Some(right) = node.child_by_field_name("right") {
        extract_dfg_python(right, tokens, states, result);
    }

    // 왼쪽 처리: left 변수들에 right 변수들을 sources로 설정
    if let Some(left) = node.child_by_field_name("left") {
        let right_idxs = collect_variable_indices(
            node.child_by_field_name("right"), tokens
        );
        let left_vars = collect_variable_indices(Some(left), tokens);

        for idx in left_vars {
            let token_text = tokens[idx].token.clone();
            result.push(DfgNode {
                name:    token_text.clone(),
                index:   idx,
                sources: right_idxs.clone(),
            });
            states.insert(token_text, vec![idx]);
        }
    }
}

// ── for statement 처리 ──────────────────────────────────────────────
fn handle_for_statement(
    node:    Node,
    tokens:  &[TokenIndex],
    states:  &mut HashMap<String, Vec<usize>>,
    result:  &mut Vec<DfgNode>,
) {
    // 이터러블 먼저 처리
    if let Some(right) = node.child_by_field_name("right") {
        extract_dfg_python(right, tokens, states, result);
    }

    // 루프 변수 정의
    if let Some(left) = node.child_by_field_name("left") {
        let right_idxs = collect_variable_indices(
            node.child_by_field_name("right"), tokens
        );
        let left_vars = collect_variable_indices(Some(left), tokens);
        for idx in left_vars {
            let token_text = tokens[idx].token.clone();
            result.push(DfgNode {
                name:    token_text.clone(),
                index:   idx,
                sources: right_idxs.clone(),
            });
            states.insert(token_text, vec![idx]);
        }
    }

    // 바디 처리
    if let Some(body) = node.child_by_field_name("body") {
        let mut cursor = body.walk();
        for child in body.children(&mut cursor) {
            extract_dfg_python(child, tokens, states, result);
        }
    }
}

// ── tree_to_variable_index (Python utils.py 동일) ───────────────────
// 변수 인덱스만 수집 (keyword/operator 제외)
fn collect_variable_indices(node: Option<Node>, tokens: &[TokenIndex]) -> Vec<usize> {
    let node = match node { Some(n) => n, None => return vec![] };
    let mut result = Vec::new();
    collect_var_idx_inner(node, tokens, &mut result);
    result
}

fn collect_var_idx_inner(node: Node, tokens: &[TokenIndex], result: &mut Vec<usize>) {
    let is_leaf   = node.child_count() == 0;
    let is_string = node.kind() == "string";
    let is_comment = node.kind() == "comment";

    if (is_leaf || is_string) && !is_comment {
        let pos  = node.start_position();
        let epos = node.end_position();
        if let Some(idx) = find_token_idx(tokens, pos.row, pos.column, epos.row, epos.column) {
            let token_text = tokens[idx].token.clone();
            let node_kind  = node.kind();
            // Python tree_to_variable_index: type != code 인 것만
            if node_kind != token_text {
                result.push(idx);
            }
        }
        return;
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_var_idx_inner(child, tokens, result);
    }
}
