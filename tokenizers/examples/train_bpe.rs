use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::normalizers::{strip::Strip, unicode::NFC, utils::Sequence, NormalizerWrapper};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::{AddedToken, Model, Result, TokenizerBuilder};

use std::path::Path;

fn main() -> Result<()> {
    let vocab_size: usize = 100;

    let min_frequency = 0;
    let add_prefix_space = false;
    let trim_offsets = false;
    let use_regex = false;

    let mut trainer = BpeTrainerBuilder::new()
        .show_progress(true)
        .vocab_size(vocab_size)
        .min_frequency(min_frequency)
        .special_tokens(vec![
            AddedToken::from(String::from("<PAD>"), true),
            AddedToken::from(String::from("<MASK_TOKEN>"), true),
            AddedToken::from(String::from("<BOS_TOKEN>"), true),
            AddedToken::from(String::from("<EOS_TOKEN>"), true),
            AddedToken::from(String::from("<UNK>"), true),
        ])
        .build();

    let mut tokenizer = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_normalizer(Some(Sequence::new(vec![])))
        .with_pre_tokenizer(Some(ByteLevel::new(add_prefix_space, trim_offsets, use_regex)))
        .with_post_processor(Some(ByteLevel::new(add_prefix_space, trim_offsets, use_regex)))
        .with_decoder(Some(ByteLevel::new(add_prefix_space, trim_offsets, use_regex)))
        .build()?;

    let pretty = false;
    tokenizer
        .train_from_pretokenized_data(
            // .train_from_files(
            &mut trainer,
            vec!["/home/felipe_cohere_com/pretokenized.tsv".to_string()],
        )?
        .save("tokenizer.json", pretty)?;

    Ok(())
}