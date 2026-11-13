// Clean-room byte-level BPE tokenizer (Qwen-family tokenizer.json format).
//
// Sources of truth: the model's own tokenizer.json (Apache-2.0) and public
// GPT-2/Qwen2 byte-level BPE semantics. Verified id-for-id against HF
// transformers (fast tokenizer) by tools/tok_ab.py.
//
// Behaviors confirmed against HF that are worth knowing:
//  - The reference is the transformers-v5 Qwen2Tokenizer, whose pre-tokenizer
//    is rebuilt from the library's class default rather than taken verbatim
//    from tokenizer.json. Both agree except that the library keeps `\p{M}`
//    out of the letter run; see the note above match_pretoken() in the .cpp.
//    tools/probe_service.py verifies by black-box prompt_tokens that the live
//    Python service follows the library, so that is what this file implements.
//  - The pre-tokenizer regex uses `\p{N}` (each digit its own pre-token), NOT
//    the classic Qwen2 `\p{N}{1,3}`. "12345" splits into five pre-tokens.
//  - The letter alternative `[^\r\n\p{L}\p{N}]?\p{L}+` allows ANY single
//    non-letter/non-digit/non-newline char (including a space, tab, or
//    combining mark) as prefix before a letter run: " abc" and "\tabc" are
//    each one pre-token, and so is "̈_s" up to the mark.
//  - `\s` follows the Rust regex crate = Unicode White_Space property
//    (25 codepoints: includes U+0085/U+00A0/U+1680/U+2000-200A/U+2028-2029/
//    U+202F/U+205F/U+3000; NOT U+001C-001F).
//  - `\s+(?!\S)` (greedy with backtrack + lookahead): a whitespace run at
//    end-of-string matches whole; a run before non-whitespace leaves its
//    last char for the next match ("  a" -> " " + " a"... actually
//    " " then " a" via the letter-prefix rule).
//  - Added (special) tokens are extracted longest-match-first from the raw
//    text before normalization/BPE, even when add_special_tokens=false
//    (that flag is a no-op here: the post-processor adds nothing).
//    encode*/exclude_added skips matching the listed added-token ids, so
//    their literal text goes through plain BPE instead (used to keep
//    user-typed <|image_pad|>/<|video_pad|> from becoming real vision
//    placeholder ids; the vision path inserts those programmatically).
//  - encode_with_offsets(..., literal) takes the client-content byte ranges
//    of a rendered chat prompt (chat_template RenderResult::literal): an
//    added token whose text overlaps such a range BPEs as ordinary text,
//    except inside a keep_plain range, where non-special added tokens
//    (<think>, <tool_call>) are still recognized. Pieces are not split at
//    range boundaries, so text without literal control strings encodes
//    exactly as without the ranges.
//  - decode(skip_special_tokens=true) drops only added tokens flagged
//    special=true; non-special added tokens like `<tool_call>` survive as
//    literal text.
//  - decode replaces ill-formed UTF-8 with U+FFFD using the Unicode
//    "maximal subpart" rule (same as Rust String::from_utf8_lossy and
//    Python bytes.decode('utf-8', 'replace')).
//  - Normalizer is NFC only: canonical equivalents compose (e + U+0301 ->
//    U+00E9), compatibility variants (halfwidth katakana, circled digits)
//    are left untouched.
//  - Unicode data tables are Unicode 15.1.0 (see unicode_tables.inc). HF's
//    regex crate may use a newer Unicode; codepoints assigned after 15.1
//    can in principle classify differently (not exercised by the A/B corpus,
//    which only samples codepoints assigned in 15.1).
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace qwenox {

struct TokenSpan {
    int id;
    size_t start;  // byte offset in the input text (inclusive)
    size_t end;    // byte offset in the input text (exclusive)
};

// A byte range of client-supplied text inside a rendered prompt. keep_plain:
// non-special added tokens stay recognized (assistant history, which holds
// the model's own <think>/<tool_call> ids decoded back to text).
struct LiteralRange {
    size_t begin;
    size_t end;
    bool keep_plain;
};

class Tokenizer {
  public:
    // Loads <dir>/tokenizer.json. Returns false on error (message in `err`).
    bool load(const std::string& dir, std::string* err = nullptr);

    // Loads the tokenizer embedded in a GGUF file's metadata
    // (tokenizer.ggml.tokens / token_type / merges). The metadata is
    // synthesized into the tokenizer.json shape and goes through the exact
    // same construction path as load(), so encode/decode behavior is
    // identical for the same model. Only the first shard is read.
    bool load_gguf(const std::string& path, std::string* err = nullptr);

    // Encode text to token ids. `add_special_tokens` is accepted for API
    // parity with HF; this model's post-processor never adds tokens, and
    // added tokens appearing literally in the text are always recognized
    // (except the ids in `exclude_added`, which BPE as ordinary text).
    std::vector<int> encode(
        const std::string& text, bool add_special_tokens = false,
        const std::unordered_set<int>* exclude_added = nullptr) const;

    // Like encode(), but also reports the byte span of each token in the
    // original input text. When NFC normalization rewrites a span, the
    // reported offsets are interpolated from the surrounding codepoints
    // (exact whenever normalization is the identity on that span).
    // `literal` (sorted, disjoint) limits where added tokens are recognized;
    // see the header notes.
    std::vector<TokenSpan> encode_with_offsets(
        const std::string& text, bool add_special_tokens = false,
        const std::unordered_set<int>* exclude_added = nullptr,
        const std::vector<LiteralRange>* literal = nullptr) const;

    // The added tokens encode_with_offsets(text, false, exclude_added,
    // literal) would produce, in order, without running BPE.
    std::vector<TokenSpan> added_matches(
        const std::string& text, const std::unordered_set<int>* exclude_added = nullptr,
        const std::vector<LiteralRange>* literal = nullptr) const;

    // Decode token ids back to text. skip_special_tokens drops added tokens
    // flagged special=true. Ill-formed UTF-8 byte runs become U+FFFD
    // (maximal-subpart rule, matching HF).
    std::string decode(const std::vector<int>& ids,
                       bool skip_special_tokens = true) const;

    // The raw byte stream the ids stand for, before UTF-8 lossy repair. This is
    // what an incremental streamer needs: a multi-byte character split across
    // tokens must be held back until it is complete, which is impossible once
    // decode() has already replaced the partial run with U+FFFD.
    std::string decode_bytes(const std::vector<int>& ids,
                             bool skip_special_tokens = true) const;

    // Look up an added token's id by literal content; -1 if not found.
    int added_token_id(const std::string& content) const;
    bool is_special_token(int id) const;
    // Any added token (special or not). encode() splits on these before BPE,
    // so the text after one encodes independently of the text before it.
    bool is_added_token(int id) const { return added_special_.count(id) != 0; }
    int vocab_size() const { return static_cast<int>(id_to_token_.size()); }
    int eos_token_id() const { return eos_id_; }   // <|im_end|>
    int pad_token_id() const { return pad_id_; }   // <|endoftext|>

  private:
    struct AddedToken {
        std::string content;
        int id;
        bool special;
    };

    // load()/load_gguf() 共用的构造路径：解析 tokenizer.json 形态的 JSON 文本。
    bool from_json_text(const std::string& text, const std::string& src,
                        std::string* err);

    std::unordered_map<std::string, int> vocab_;      // token str -> id
    std::vector<std::string> id_to_token_;            // id -> token str
    // Merge table: (left_id, right_id) -> (rank << 32) | merged_id.
    std::unordered_map<uint64_t, uint64_t> merges_;
    std::vector<AddedToken> added_;                   // sorted longest-first
    bool added_first_[256] = {};                      // first bytes of added_
    std::unordered_map<int, bool> added_special_;     // id -> special flag
    std::unordered_map<std::string, int> added_ids_;  // content -> id
    int eos_id_ = -1;
    int pad_id_ = -1;

    void bpe(const std::string& pretoken_alphabet, std::vector<int>* out,
             std::vector<uint32_t>* token_char_lens) const;
    // The added token encode recognizes at text[pos], or nullptr.
    const AddedToken* match_added(const std::string& text, size_t pos,
                                  const std::unordered_set<int>* exclude_added,
                                  const std::vector<LiteralRange>* literal) const;
};

}  // namespace qwenox
