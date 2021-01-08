use ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

pub fn merge(
    attention_in: ArrayView2<f32>,
    tokens: Vec<&str>,
    words: Vec<&str>,
    word_ends: Vec<&str>,
) -> Array2<f32> {
    assert_eq!(attention_in.len(), tokens.len() * tokens.len());
    assert!(words.len() <= tokens.len());
    assert!(words.len() == word_ends.len());

    let mut merged_attention = Array2::<f32>::zeros((tokens.len(), words.len()));

    for (token_i, _) in tokens.iter().enumerate() {
        let mut attention_sum = 0.0;
        let mut word_j = 0;
        for (token_j, token_to) in tokens.iter().enumerate() {
            attention_sum += attention_in[[token_i, token_j]];
            if word_ends[word_j..].contains(token_to) {
                word_j = match word_ends[word_j..].iter().position(|end| end == token_to) {
                    None => panic!("Just checked that token_to was in word_ends!"),
                    Some(i) => i,
                };
                merged_attention[[token_i, word_j]] = attention_sum;
                attention_sum = 0.0;
            }
        }
    }

    let mut final_attention = Array2::zeros((words.len(), words.len()));

    for (word_j, _) in words.iter().enumerate() {
        let mut word_i = 0;
        let mut attention_to_word = 0.0;
        let mut tokens_to_word_count = 0;

        for (token_i, token) in tokens.iter().enumerate() {
            attention_to_word += merged_attention[[token_i, word_j]];
            tokens_to_word_count += 1;

            if word_ends[word_i..].contains(token) {
                word_i = match word_ends[word_i..].iter().position(|end| end == token) {
                    None => panic!("every word end must be contained in tokens!"),
                    Some(i) => i,
                };

                let attention_from_word = attention_to_word / tokens_to_word_count as f32;
                final_attention[[word_i, word_j]] = attention_from_word;
                attention_to_word = 0.0;
                tokens_to_word_count = 0;
            }
        }
    }

    final_attention
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn simple_two_words_merge() {
        let tokens = vec!["A", "B"];
        let words = vec!["AB"];
        let word_ends = vec!["B"];
        let attention = arr2(&[[0.2, 0.8], [0.2, 0.8]]);
        let merged = merge(attention.view(), tokens, words, word_ends);
        assert_eq!(merged, Array2::ones((1, 1)));
    }

    #[test]
    fn simple_two_words_no_merge() {
        let tokens = vec!["A", "B"];
        let words = vec!["A", "B"];
        let word_ends = vec!["A", "B"];
        let attention = arr2(&[[0.2, 0.8], [0.2, 0.8]]);
        let merged = merge(attention.view(), tokens, words, word_ends);
        assert_eq!(merged, attention);
    }
}

#[pymodule]
fn merge_attention(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "merge")]
    fn merge_py<'py>(
        py: Python<'py>,
        attention_in: PyReadonlyArray2<f32>,
        tokens: Vec<&str>,
        words: Vec<&str>,
        word_ends: Vec<&str>,
    ) -> &'py PyArray2<f32> {
        let attention_in = attention_in.as_array();
        merge(attention_in, tokens, words, word_ends).into_pyarray(py)
    }

    Ok(())
}
