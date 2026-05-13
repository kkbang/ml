mod extractor;
mod python_dfg;
mod lang_dfg;

use pyo3::prelude::*;
use tree_sitter::{Parser, Language};
use extractor::{tree_to_token_index, DfgNode};
use std::collections::HashMap;
use libloading::{Library, Symbol};
use std::sync::OnceLock;

static SO_LIB: OnceLock<Library> = OnceLock::new();
const SO_PATH: &str = "/home/ngseokim/code-killr/parser/my-languages.so";

fn get_language(language: &str) -> Option<Language> {
    let lib = SO_LIB.get_or_init(|| {
        unsafe { Library::new(SO_PATH).expect("my-languages.so 로드 실패") }
    });
    let sym_name = format!("tree_sitter_{}\0", language);
    unsafe {
        let func: Result<Symbol<unsafe fn() -> Language>, _> = lib.get(sym_name.as_bytes());
        func.ok().map(|f| f())
    }
}

#[pyfunction]
fn extract_dataflow_rust(
    _py: Python, code: &str, language: &str,
) -> PyResult<(Vec<String>, Vec<(String, usize, Vec<usize>)>)> {
    let preprocessed;
    let code_to_parse = if language == "php" {
        preprocessed = format!("<?php {}", code);
        &preprocessed[..]
    } else { code };

    let ts_lang = get_language(language)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
            format!("지원하지 않는 언어: {}", language)))?;

    let mut parser = Parser::new();
    parser.set_language(ts_lang)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let tree = parser.parse(code_to_parse, None)
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("파싱 실패"))?;

    let lines: Vec<&str> = code_to_parse.lines().collect();
    let mut token_index = Vec::new();
    tree_to_token_index(tree.root_node(), &lines, &mut token_index);

    if token_index.is_empty() { return Ok((vec![], vec![])); }

    let code_tokens: Vec<String> = token_index.iter().map(|t| t.token.clone()).collect();
    let mut states: HashMap<String, Vec<usize>> = HashMap::new();
    let mut dfg_nodes: Vec<DfgNode> = Vec::new();

    match language {
        "python" => {
            python_dfg::extract_dfg_python(
                tree.root_node(), &token_index, &mut states, &mut dfg_nodes,
            );
        }
        "java" => {
            lang_dfg::extract(tree.root_node(), &token_index, &mut states, &mut dfg_nodes, &lang_dfg::JAVA);
        }
        "c_sharp" => {
            lang_dfg::extract(tree.root_node(), &token_index, &mut states, &mut dfg_nodes, &lang_dfg::CSHARP);
        }
        "ruby" => {
            lang_dfg::extract(tree.root_node(), &token_index, &mut states, &mut dfg_nodes, &lang_dfg::RUBY);
        }
        "go" => {
            lang_dfg::extract(tree.root_node(), &token_index, &mut states, &mut dfg_nodes, &lang_dfg::GO);
        }
        "php" => {
            lang_dfg::extract(tree.root_node(), &token_index, &mut states, &mut dfg_nodes, &lang_dfg::PHP);
        }
        "javascript" => {
            lang_dfg::extract(tree.root_node(), &token_index, &mut states, &mut dfg_nodes, &lang_dfg::JS);
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("지원하지 않는 언어: {}", language)));
        }
    }

    dfg_nodes.sort_by_key(|n| n.index);
    dfg_nodes.retain(|n| n.index < code_tokens.len());
    let dfg = dfg_nodes.into_iter().map(|n| (n.name, n.index, n.sources)).collect();
    Ok((code_tokens, dfg))
}

#[pymodule]
fn dfg_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_dataflow_rust, m)?)?;
    Ok(())
}
