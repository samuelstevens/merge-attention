use ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// Checks if there are any duplicated word_ends that are not in words
pub fn split_precondition(words: &Vec<&str>, word_ends: &Vec<&str>) -> bool {
    let mut duplicated_word_ends = vec![];

    for (end1, end2) in word_ends.iter().zip(word_ends[1..].iter()) {
        if end1 == end2 {
            duplicated_word_ends.push(end1);
        }
    }

    if duplicated_word_ends.is_empty() {
        return false;
    }

    let mut duplicate_not_in_word = false;
    for duplicate in duplicated_word_ends.iter() {
        if !words.contains(duplicate) {
            duplicate_not_in_word = true;
            break;
        }
    }

    duplicate_not_in_word
}

pub fn merge(
    attention_in: ArrayView2<f32>,
    tokens: Vec<&str>,
    words: Vec<&str>,
    word_ends: Vec<&str>,
) -> Array2<f32> {
    assert_eq!(attention_in.len(), tokens.len() * tokens.len());

    if !split_precondition(&words, &word_ends) {
        assert!(words.len() <= tokens.len());
        assert!(words.len() == word_ends.len());
    }

    let mut merged_attention = Array2::<f32>::zeros((tokens.len(), words.len()));

    if word_ends.len() == 0 {
        return merged_attention + attention_in;
    }

    for (token_i, _) in tokens.iter().enumerate() {
        let mut attention_sum = 0.0;
        let mut word_j = 0;
        for (token_j, token_to) in tokens.iter().enumerate() {
            attention_sum += attention_in[[token_i, token_j]];
            // println!("{} {}", word_ends[word_j], token_to);
            if *token_to == word_ends[word_j] {
                merged_attention[[token_i, word_j]] = attention_sum;
                attention_sum = 0.0;

                word_j += 1;
                while word_j < word_ends.len() && *word_ends[word_j - 1] == *word_ends[word_j] {
                    word_j += 1;
                }
            }
        }
    }

    // println!("{:?}", merged_attention);

    let mut final_attention = Array2::zeros((words.len(), words.len()));

    for (word_j, _) in words.iter().enumerate() {
        let mut word_i = 0;
        let mut attention_to_word = 0.0;
        let mut tokens_to_word_count = 0;

        for (token_i, token) in tokens.iter().enumerate() {
            attention_to_word += merged_attention[[token_i, word_j]];
            tokens_to_word_count += 1;
            // println!("{} {}", word_ends[word_i], token);
            if *token == word_ends[word_i] {
                final_attention[[word_i, word_j]] = attention_to_word / tokens_to_word_count as f32;
                attention_to_word = 0.0;
                tokens_to_word_count = 0;
                word_i += 1;
                // TODO: turn this check into a function that makes it clear what I'm checking
                // (I'm checking if any word_ends are duplicated, like ")," "),")
                // TODO: Does this break with examples that genuinely have two duplicates
                while word_i < word_ends.len() && *word_ends[word_i - 1] == *word_ends[word_i] {
                    word_i += 1;
                }
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

    #[test]
    fn three_by_three() {
        let tokens = vec!["A", "B", "C"];
        let words = vec!["A", "B", "C"];
        let word_ends = vec!["A", "B", "C"];
        let attention = Array2::<f32>::ones((3, 3));
        let merged = merge(attention.view(), tokens, words, word_ends);
        assert_eq!(merged, attention);
    }

    #[test]
    fn first_word_merge() {
        let tokens = vec!["A", "B", "C"];
        let words = vec!["AB", "C"];
        let word_ends = vec!["B", "C"];
        let attention = Array2::<f32>::ones((3, 3));
        let merged = merge(attention.view(), tokens, words, word_ends);
        let expected = arr2(&[[2., 1.], [2., 1.]]);
        assert_eq!(merged, expected);
    }

    #[test]
    fn repeats() {
        let tokens = vec!["A", "B", "A"];
        let words = vec!["AB", "A"];
        let word_ends = vec!["B", "A"];
        let attention = Array2::<f32>::ones((3, 3));
        let merged = merge(attention.view(), tokens, words, word_ends);
        let expected = arr2(&[[2., 1.], [2., 1.]]);
        assert_eq!(merged, expected);
    }

    #[test]
    fn repeat_word_ends() {
        let tokens = vec!["a", "bc", "d", "e"];
        let words = vec!["a", "b", "c", "de"];
        let word_ends = vec!["a", "bc", "bc", "e"];
        let attention = Array2::<f32>::ones((4, 4));
        let merged = merge(attention.view(), tokens, words, word_ends);
        let expected = arr2(&[
            [1., 1., 0., 2.],
            [1., 1., 0., 2.],
            [0., 0., 0., 0.],
            [1., 1., 0., 2.],
        ]);
        assert_eq!(merged, expected);
    }

    #[test]
    fn precondition() {
        let words = vec!["A", "B"];
        let word_ends = vec!["AB", "AB"];
        assert!(split_precondition(&words, &word_ends));
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
