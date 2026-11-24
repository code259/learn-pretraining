"""Run with:  python test_bpe.py     (no dependencies)
Each test prints PASS/FAIL with a hint. Work top to bottom."""
import sys, traceback
from bpe import get_pair_counts, merge_word, train_from_counts, train_bpe, encode, decode, pretokenize

def t1_pair_counts():
    words = {(b"l", b"o", b"w"): 5, (b"l", b"o", b"w", b"e", b"r"): 2}
    c = get_pair_counts(words)
    assert c[(b"l", b"o")] == 7, f"(l,o) should be 7, got {c[(b'l', b'o')]}"
    assert c[(b"e", b"r")] == 2
    assert (b"w", b"l") not in c or c[(b"w", b"l")] == 0, "pairs never cross word boundaries"

def t2_merge_word():
    assert merge_word((b"n", b"e", b"w", b"e", b"s", b"t"), (b"s", b"t")) == (b"n", b"e", b"w", b"e", b"st")
    assert merge_word((b"a", b"a", b"a"), (b"a", b"a")) == (b"aa", b"a"), "merge left-to-right, non-overlapping"
    assert merge_word((b"a", b"b"), (b"x", b"y")) == (b"a", b"b")

def t3_sennrich_example():
    counts = {"low": 5, "lower": 2, "widest": 3, "newest": 6}
    merges = train_from_counts(counts, num_merges=6)
    got = [(a + b).decode() for a, b in merges]
    assert got[:4] == ["st", "est", "ow", "low"], f"first 4 merges should be st, est, ow, low; got {got[:4]} (tie-break: lexicographically greater pair)"
    assert got == ["st", "est", "ow", "low", "west", "ne"], f"got {got}"

def t4_pretokenize():
    assert pretokenize("The cat sat.") == ["The", " cat", " sat", "."]
    assert pretokenize("in 2026!") == ["in", " ", "2", "0", "2", "6", "!"], "digits split individually, like SmolLM2"

def t5_train_and_roundtrip():
    text = open(__file__, encoding="utf-8").read() * 3 + " naïve café 🤖 "
    vocab, merges = train_bpe(text, vocab_size=400)
    assert len(vocab) == 400 and len(merges) == 400 - 256, f"vocab {len(vocab)}, merges {len(merges)}"
    assert all(vocab[i] == bytes([i]) for i in range(256)), "ids 0..255 must be the raw bytes"
    for s in ["hello world", "naïve café 🤖", "def t1_pair_counts():", ""]:
        ids = encode(s, merges)
        assert decode(ids, vocab) == s, f"round trip failed on {s!r}"

def t6_encode_uses_merges():
    text = "the theme then there " * 50
    vocab, merges = train_bpe(text, vocab_size=270)
    ids = encode(" the", merges)
    assert len(ids) == 1, f"' the' is frequent; it should be a single token, got {[vocab[i] for i in ids]}"
    assert len(encode(text, merges)) < len(text.encode()) / 2, "expected >2x compression on training text"

tests = [t1_pair_counts, t2_merge_word, t3_sennrich_example, t4_pretokenize, t5_train_and_roundtrip, t6_encode_uses_merges]
ok = 0
for t in tests:
    try:
        t(); print(f"PASS  {t.__name__}"); ok += 1
    except NotImplementedError:
        print(f"TODO  {t.__name__}")
    except Exception as e:
        print(f"FAIL  {t.__name__}: {e}")
        if "-v" in sys.argv: traceback.print_exc()
print(f"\n{ok}/{len(tests)} passing")
