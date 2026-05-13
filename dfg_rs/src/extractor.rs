// extractor.rs
// GraphCodeBERT DFG 형식에 맞는 공통 데이터 구조
//
// Python 원본 DFG 형식:
//   dfg entry = (var_name, token_idx, usage_type, [source_indices], ...)
//   dfg_to_code = [(start, end), ...]  ← DFG 노드 → 코드 토큰 위치
//   dfg_to_dfg  = [[idx, ...], ...]    ← DFG 노드 → DFG 노드 엣지

use std::collections::HashMap;

/// 단일 DFG 노드
/// Python 원본: (name, index, type, [sources])
#[derive(Debug, Clone)]
pub struct DfgNode {
    pub name:    String,         // 변수명
    pub index:   usize,          // 코드 토큰에서의 위치 인덱스
    pub sources: Vec<usize>,     // 이 변수가 값을 받아오는 소스 인덱스들
}

/// 전체 DFG 추출 결과
#[derive(Debug)]
pub struct DfgResult {
    pub code_tokens: Vec<String>,        // 파싱된 코드 토큰들
    pub dfg:         Vec<DfgNode>,       // DFG 노드 리스트
}

/// AST 노드의 바이트 범위 → 코드 토큰 인덱스 매핑
/// Python 원본의 index_to_code 딕셔너리에 해당
#[derive(Debug, Clone)]
pub struct TokenIndex {
    pub start_row: usize,
    pub start_col: usize,
    pub end_row:   usize,
    pub end_col:   usize,
    pub token_idx: usize,   // code_tokens에서의 인덱스
    pub token:     String,
}

/// 코드를 줄 단위로 분리
pub fn split_lines(code: &str) -> Vec<&str> {
    code.lines().collect()
}

/// (row, col) 범위에서 실제 토큰 문자열 추출
pub fn extract_token(lines: &[&str], start: (usize, usize), end: (usize, usize)) -> String {
    let (sr, sc) = start;
    let (er, ec) = end;

    if sr == er {
        // 한 줄 안에 있는 토큰
        if let Some(line) = lines.get(sr) {
            let chars: Vec<char> = line.chars().collect();
            let sc = sc.min(chars.len());
            let ec = ec.min(chars.len());
            return chars[sc..ec].iter().collect();
        }
    }
    // 여러 줄에 걸친 토큰 (거의 없음)
    String::new()
}

/// tree_to_token_index — Python 원본과 동일한 로직
/// Python 조건:
///   1. 단일 라인 + 내용 있음 → 이 노드 자체를 토큰으로 반환 (자식 무시)
///   2. 자식 없음 → 빈 결과 (스킵)
///   3. 자식 있음 → 재귀
/// tree_to_token_index — Python utils.py와 동일한 로직
/// Python:
///   if (leaf OR string) AND not comment → include this node
///   else → recurse into children
pub fn tree_to_token_index(
    node:   tree_sitter::Node,
    lines:  &[&str],
    result: &mut Vec<TokenIndex>,
) {
    let is_leaf    = node.child_count() == 0;
    let is_string  = node.kind() == "string";
    let is_comment = node.kind() == "comment";

    if (is_leaf || is_string) && !is_comment {
        let start = node.start_position();
        let end   = node.end_position();
        let token = extract_token(lines, (start.row, start.column), (end.row, end.column));
        // Python은 빈 문자열도 포함 (empty token 필터링 없음)
        result.push(TokenIndex {
            start_row: start.row, start_col: start.column,
            end_row:   end.row,   end_col:   end.column,
            token_idx: result.len(),
            token,
        });
        return;  // string 노드는 자식으로 재귀 안 함
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        tree_to_token_index(child, lines, result);
    }
}

/// 노드의 (row, col) 범위로 token_idx 찾기
/// Python 원본의 index_to_code 역방향 매핑에 해당
pub fn find_token_idx(
    tokens: &[TokenIndex],
    start_row: usize,
    start_col: usize,
    end_row:   usize,
    end_col:   usize,
) -> Option<usize> {
    tokens.iter().position(|t| {
        t.start_row == start_row
            && t.start_col == start_col
            && t.end_row   == end_row
            && t.end_col   == end_col
    })
}

/// DFG 결과를 GraphCodeBERT encode_with_dfg 형식으로 변환
/// Python 원본의 dfg_to_code, dfg_to_dfg 생성에 해당
pub fn build_dfg_mappings(
    dfg:    &[DfgNode],
    tokens: &[TokenIndex],
) -> (Vec<(usize, usize)>, Vec<Vec<usize>>) {
    // dfg_to_code: 각 DFG 노드가 대응하는 코드 토큰 범위 (start, end)
    // Python 원본: ori2cur_pos[x[1]] where x[1] = token_idx
    let dfg_to_code: Vec<(usize, usize)> = dfg.iter().map(|node| {
        let idx = node.index;
        // token_idx → (idx+1, idx+1) 형태의 단일 토큰 범위
        // Python 원본과 동일: ori2cur_pos는 서브토큰 범위를 반환
        let start = idx;
        let end   = idx + 1;
        (start, end)
    }).collect();

    // dfg_to_dfg: 각 DFG 노드의 소스 DFG 노드 인덱스들
    // Python 원본의 reverse_index 매핑 적용
    let mut reverse_index: HashMap<usize, usize> = HashMap::new();
    for (i, node) in dfg.iter().enumerate() {
        reverse_index.insert(node.index, i);
    }

    let dfg_to_dfg: Vec<Vec<usize>> = dfg.iter().map(|node| {
        node.sources.iter()
            .filter_map(|&src| reverse_index.get(&src).copied())
            .collect()
    }).collect();

    (dfg_to_code, dfg_to_dfg)
}
