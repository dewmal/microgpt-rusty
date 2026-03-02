mod data;
mod math;
mod model;
mod tokenizer;
mod value;

fn main() {
    data::load_data();
    tokenizer::tokenizer();

    let vocab_size = 27;
    let block_size = 8;
    let n_layer = 1;
    let n_embed = 16;

    let model = model::Model::new(vocab_size, block_size, n_layer, n_embed);
    println!("Num params: {}", model.params.len());
}
