use crate::data::preprocess_data;

pub(crate) fn tokenizer() {
    let docs = preprocess_data();
    let bos = docs.len();
    let vocab_size = docs.len() + 1;
    println!("BOS:{bos},vocab size: {vocab_size}")
}
