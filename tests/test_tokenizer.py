"""Phase 3 tests: CorpusTokenizer JSON workflow and append materialization.

Tests cover:
- CorpusTokenizer.create_new() with special tokens in fixed order
- JSON round-trip (to_json / from_json)
- Append compatibility: namespace match/mismatch, duplicate token rejection
- Token ID stability across appends
- tokenize_labels() translation
- create_new and append_routed route factory
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from perturb_data_lab.materializers import (
    CorpusTokenizer,
    update_corpus_index,
)
from perturb_data_lab.materializers.models import (
    CountSourceSpec,
    DatasetJoinRecord,
    GlobalMetadataDocument,
    MaterializationManifest,
    OutputRoots,
    ProvenanceSpec,
)


# ---------------------------------------------------------------------------
# CorpusTokenizer.create_new
# ---------------------------------------------------------------------------


class TestCorpusTokenizerCreateNew:
    def test_special_tokens_fixed_ids(self):
        """Special tokens <pad>, <cls>, <unk>, <eos> always receive IDs 0-3."""
        tok = CorpusTokenizer.create_new(
            corpus_id="test-corpus",
            namespace="gene_symbol",
            regular_tokens=["BRCA1", "TP53", "EGFR"],
        )
        assert tok.to_id("<pad>") == 0
        assert tok.to_id("<cls>") == 1
        assert tok.to_id("<unk>") == 2
        assert tok.to_id("<eos>") == 3

    def test_regular_tokens_sorted_ascending(self):
        """Regular tokens receive IDs in sorted ascending order after special tokens."""
        tok = CorpusTokenizer.create_new(
            corpus_id="test-corpus",
            namespace="gene_symbol",
            regular_tokens=["TP53", "BRCA1", "EGFR"],
        )
        # Sorted: BRCA1 < EGFR < TP53
        assert tok.to_id("BRCA1") == 4  # first regular
        assert tok.to_id("EGFR") == 5
        assert tok.to_id("TP53") == 6

    def test_empty_regular_tokens(self):
        """Tokenizer with no regular tokens still has special tokens."""
        tok = CorpusTokenizer.create_new(
            corpus_id="empty-corpus",
            namespace="gene_symbol",
            regular_tokens=[],
        )
        assert tok.n_tokens == 4
        assert tok.to_id("<pad>") == 0
        assert tok.max_id == 3

    def test_duplicate_regular_tokens_removed(self):
        """Duplicate tokens in the input list are deduplicated."""
        tok = CorpusTokenizer.create_new(
            corpus_id="test-corpus",
            namespace="gene_symbol",
            regular_tokens=["TP53", "TP53", "BRCA1"],
        )
        # Only unique tokens, still sorted
        assert tok.to_id("BRCA1") == 4
        assert tok.to_id("TP53") == 5
        assert tok.n_tokens == 6  # 4 special + 2 unique regular


# ---------------------------------------------------------------------------
# CorpusTokenizer JSON round-trip
# ---------------------------------------------------------------------------


class TestCorpusTokenizerJsonRoundTrip:
    def test_round_trip_preserves_stoi(self):
        """to_json / from_json round-trip preserves all stoi entries."""
        tok = CorpusTokenizer.create_new(
            corpus_id="round-trip-test",
            namespace="ensembl",
            regular_tokens=["ENSG00000139618", "ENSG00000157764"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tokenizer.json"
            tok.to_json(path)
            loaded = CorpusTokenizer.from_json(path)

        assert loaded.corpus_id == tok.corpus_id
        assert loaded.namespace == tok.namespace
        assert loaded.contract_version == tok.contract_version
        assert loaded.special_tokens == ("<pad>", "<cls>", "<unk>", "<eos>")
        # All original tokens preserved
        assert loaded.to_id("ENSG00000139618") == tok.to_id("ENSG00000139618")
        assert loaded.to_id("ENSG00000157764") == tok.to_id("ENSG00000157764")
        assert loaded.to_id("<pad>") == 0
        # Same total count
        assert loaded.n_tokens == tok.n_tokens

    def test_to_json_file_created(self):
        """to_json creates the output file."""
        tok = CorpusTokenizer.create_new(
            corpus_id="file-test",
            namespace="gene_symbol",
            regular_tokens=["GENE1"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "tokenizer.json"
            tok.to_json(path)
            assert path.exists()
            # Verify it's valid JSON
            import json
            with open(path) as fh:
                data = json.load(fh)
            assert "stoi" in data
            assert data["corpus_id"] == "file-test"


# ---------------------------------------------------------------------------
# CorpusTokenizer append compatibility
# ---------------------------------------------------------------------------


class TestCorpusTokenizerAppendCompatibility:
    def test_namespace_mismatch_rejected(self):
        """Append with wrong namespace is rejected before any I/O."""
        tok = CorpusTokenizer.create_new(
            corpus_id="test-corpus",
            namespace="gene_symbol",
            regular_tokens=["BRCA1"],
        )
        compatible, reason = tok.append_compatible(["EGFR"], append_namespace="ensembl")
        assert compatible is False
        assert "namespace mismatch" in reason
        assert "gene_symbol" in reason
        assert "ensembl" in reason

    def test_namespace_match_accepted(self):
        """Append with matching namespace passes the namespace check."""
        tok = CorpusTokenizer.create_new(
            corpus_id="test-corpus",
            namespace="gene_symbol",
            regular_tokens=["BRCA1"],
        )
        compatible, reason = tok.append_compatible(["EGFR"], append_namespace="gene_symbol")
        assert compatible is True
        assert reason == ""

    def test_duplicate_tokens_accepted(self):
        """Appending tokens that already exist is accepted — they just reuse existing IDs."""
        tok = CorpusTokenizer.create_new(
            corpus_id="test-corpus",
            namespace="gene_symbol",
            regular_tokens=["BRCA1", "TP53"],
        )
        # Overlapping labels are fine — namespace match is all that matters
        compatible, reason = tok.append_compatible(
            ["BRCA1", "NEW-GENE"], append_namespace="gene_symbol"
        )
        assert compatible is True
        assert reason == ""

    def test_all_new_tokens_accepted(self):
        """Appending only new tokens passes."""
        tok = CorpusTokenizer.create_new(
            corpus_id="test-corpus",
            namespace="gene_symbol",
            regular_tokens=["BRCA1"],
        )
        compatible, reason = tok.append_compatible(
            ["TP53", "EGFR"], append_namespace="gene_symbol"
        )
        assert compatible is True


# ---------------------------------------------------------------------------
# CorpusTokenizer.append_tokens
# ---------------------------------------------------------------------------


class TestCorpusTokenizerAppendTokens:
    def test_existing_ids_unchanged(self):
        """append_tokens does not change existing token IDs."""
        tok = CorpusTokenizer.create_new(
            corpus_id="test-corpus",
            namespace="gene_symbol",
            regular_tokens=["BRCA1", "TP53"],
        )
        # Save original IDs
        brca1_id = tok.to_id("BRCA1")
        tp53_id = tok.to_id("TP53")
        pad_id = tok.to_id("<pad>")

        updated = tok.append_tokens(["EGFR", "MYC"], append_namespace="gene_symbol")

        # Original IDs unchanged
        assert updated.to_id("BRCA1") == brca1_id
        assert updated.to_id("TP53") == tp53_id
        assert updated.to_id("<pad>") == pad_id

    def test_new_tokens_appended_in_sorted_order(self):
        """New regular tokens are appended in sorted ascending order."""
        tok = CorpusTokenizer.create_new(
            corpus_id="test-corpus",
            namespace="gene_symbol",
            regular_tokens=["BRCA1"],  # ID 4
        )
        updated = tok.append_tokens(["TP53", "EGFR"], append_namespace="gene_symbol")

        # BRCA1 keeps ID 4
        assert updated.to_id("BRCA1") == 4
        # New tokens get next IDs in sorted order: EGFR(5), TP53(6)
        assert updated.to_id("EGFR") == 5
        assert updated.to_id("TP53") == 6
        assert updated.max_id == 6

    def test_append_tokens_rejects_namespace_mismatch(self):
        """append_tokens raises ValueError on namespace mismatch."""
        tok = CorpusTokenizer.create_new(
            corpus_id="test-corpus",
            namespace="gene_symbol",
            regular_tokens=["BRCA1"],
        )
        with pytest.raises(ValueError, match="namespace mismatch"):
            tok.append_tokens(["EGFR"], append_namespace="ensembl")

    def test_append_tokens_reuses_existing_ids(self):
        """append_tokens with duplicate labels reuses existing token IDs without error."""
        tok = CorpusTokenizer.create_new(
            corpus_id="test-corpus",
            namespace="gene_symbol",
            regular_tokens=["BRCA1"],
        )
        brca1_id = tok.to_id("BRCA1")
        # Appending a token that already exists is fine (idempotent)
        updated = tok.append_tokens(["BRCA1", "NEW-GENE"], append_namespace="gene_symbol")
        # Existing token ID preserved
        assert updated.to_id("BRCA1") == brca1_id
        # New token added with next available ID
        assert updated.to_id("NEW-GENE") > brca1_id

    def test_multiple_append_rounds_idempotent(self):
        """Multiple append rounds preserve stability."""
        tok = CorpusTokenizer.create_new(
            corpus_id="test-corpus",
            namespace="gene_symbol",
            regular_tokens=["A"],
        )
        a_id = tok.to_id("A")
        tok = tok.append_tokens(["B"], append_namespace="gene_symbol")
        tok = tok.append_tokens(["C"], append_namespace="gene_symbol")
        tok = tok.append_tokens(["D"], append_namespace="gene_symbol")

        # A's ID never changed
        assert tok.to_id("A") == a_id
        assert tok.to_id("B") == a_id + 1
        assert tok.to_id("C") == a_id + 2
        assert tok.to_id("D") == a_id + 3

    def test_three_step_append_overlap_reuses_existing_ids(self):
        """Three-step append (Marson → Pancreatic → XORION pattern) reuses overlapping IDs."""
        # Step 1: initial corpus with genes A, B, C
        tok = CorpusTokenizer.create_new(
            corpus_id="three-dataset-corpus",
            namespace="gene_symbol",
            regular_tokens=["A", "B", "C"],
        )
        assert tok.to_id("A") == 4
        assert tok.to_id("B") == 5
        assert tok.to_id("C") == 6

        # Step 2: append with overlap — B, C, D (B and C overlap, D is novel)
        tok = tok.append_tokens(["B", "C", "D"], append_namespace="gene_symbol")
        # A, B, C keep original IDs; D gets next ID
        assert tok.to_id("A") == 4
        assert tok.to_id("B") == 5
        assert tok.to_id("C") == 6
        assert tok.to_id("D") == 7

        # Step 3: append with more overlap — B, C, D, E, F (E, F novel)
        tok = tok.append_tokens(["B", "C", "D", "E", "F"], append_namespace="gene_symbol")
        assert tok.to_id("A") == 4
        assert tok.to_id("B") == 5
        assert tok.to_id("C") == 6
        assert tok.to_id("D") == 7
        assert tok.to_id("E") == 8
        assert tok.to_id("F") == 9
        # No new IDs assigned for the already-seen B, C, D
        assert tok.n_tokens == 10  # 4 special + 6 unique regular


# ---------------------------------------------------------------------------
# CorpusTokenizer.tokenize_labels
# ---------------------------------------------------------------------------


class TestCorpusTokenizerTokenizeLabels:
    def test_translates_labels_to_ids(self):
        """tokenize_labels translates label strings to token IDs."""
        tok = CorpusTokenizer.create_new(
            corpus_id="test-corpus",
            namespace="gene_symbol",
            regular_tokens=["BRCA1", "TP53"],
        )
        ids = tok.tokenize_labels(["BRCA1", "TP53", "<pad>"])
        assert ids == [tok.to_id("BRCA1"), tok.to_id("TP53"), tok.to_id("<pad>")]

    def test_tokenize_labels_unknown_raises(self):
        """Unknown token raises ValueError with on_unknown='raise'."""
        tok = CorpusTokenizer.create_new(
            corpus_id="test-corpus",
            namespace="gene_symbol",
            regular_tokens=["BRCA1"],
        )
        with pytest.raises(ValueError, match="not found"):
            tok.tokenize_labels(["UNKNOWN"], on_unknown="raise")

    def test_tokenize_labels_skip_returns_minus_one(self):
        """on_unknown='skip' returns -1 for unknown tokens."""
        tok = CorpusTokenizer.create_new(
            corpus_id="test-corpus",
            namespace="gene_symbol",
            regular_tokens=["BRCA1"],
        )
        ids = tok.tokenize_labels(["BRCA1", "UNKNOWN"], on_unknown="skip")
        assert ids == [tok.to_id("BRCA1"), -1]


# ---------------------------------------------------------------------------
# update_corpus_index with tokenizer_path
# ---------------------------------------------------------------------------


class TestUpdateCorpusIndexWithTokenizer:
    def test_creates_global_metadata_with_tokenizer_path(self):
        """New corpus creation writes global-metadata.yaml with tokenizer_path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_dir = Path(tmpdir) / "corpus"
            corpus_dir.mkdir()
            idx_path = corpus_dir / "corpus-index.yaml"
            global_meta = GlobalMetadataDocument(
                kind="global-metadata",
                contract_version="0.2.0",
                schema_version="0.1.0",
                feature_registry_id="",
                missing_value_literal="NA",
                raw_field_policy="preserve-unchanged",
            )

            record = DatasetJoinRecord(
                dataset_id="ds_001",
                release_id="v0.1",
                join_mode="create_new",
                manifest_path="/meta/v0.1-manifest.yaml",
            )
            # tokenizer_path is set via GlobalMetadataDocument in Phase 3 architecture
            global_meta_with_tokenizer = GlobalMetadataDocument(
                kind="global-metadata",
                contract_version="0.2.0",
                schema_version="0.1.0",
                feature_registry_id="",
                missing_value_literal="NA",
                raw_field_policy="preserve-unchanged",
                tokenizer_path="tokenizer.json",
            )
            update_corpus_index(
                idx_path,
                record,
                global_metadata=global_meta_with_tokenizer,
            )

            # global-metadata.yaml was written
            global_meta_path = corpus_dir / "global-metadata.yaml"
            assert global_meta_path.exists()

            loaded = GlobalMetadataDocument.from_yaml_file(global_meta_path)
            assert loaded.tokenizer_path == "tokenizer.json"
            assert loaded.contract_version == "0.2.0"

    def test_appending_does_not_overwrite_global_metadata(self):
        """Append to existing corpus does not rewrite global-metadata.yaml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_dir = Path(tmpdir) / "corpus"
            corpus_dir.mkdir()

            # Create initial corpus
            idx_path = corpus_dir / "corpus-index.yaml"
            global_meta = GlobalMetadataDocument(
                kind="global-metadata",
                contract_version="0.2.0",
                schema_version="0.1.0",
                feature_registry_id="",
                missing_value_literal="NA",
                raw_field_policy="preserve-unchanged",
                tokenizer_path="tokenizer.json",
            )
            record1 = DatasetJoinRecord(
                dataset_id="ds_001",
                release_id="v0.1",
                join_mode="create_new",
                manifest_path="/meta/v0.1.yaml",
            )
            update_corpus_index(idx_path, record1, global_metadata=global_meta)

            # Append second dataset
            record2 = DatasetJoinRecord(
                dataset_id="ds_002",
                release_id="v0.2",
                join_mode="append_routed",
                manifest_path="/meta/v0.2.yaml",
            )
            update_corpus_index(idx_path, record2, global_metadata=None)

            # global-metadata.yaml still exists with original content
            global_meta_path = corpus_dir / "global-metadata.yaml"
            loaded = GlobalMetadataDocument.from_yaml_file(global_meta_path)
            assert loaded.tokenizer_path == "tokenizer.json"


# ---------------------------------------------------------------------------
# MaterializationManifest tokenizer_path field
# ---------------------------------------------------------------------------


class TestMaterializationManifestTokenizerPath:
    """Tokenization is removed from MaterializationManifest in Phase 3."""

    def test_manifest_does_not_record_tokenizer_path(self):
        """MaterializationManifest.to_dict does not include tokenizer_path (Phase 3)."""
        manifest = MaterializationManifest(
            kind="materialization-manifest",
            contract_version="0.2.0",
            dataset_id="test_ds",
            release_id="v0.1",
            route="create_new",
            backend="arrow-hf",
            count_source=CountSourceSpec(selected=".X", integer_only=True),
            outputs=OutputRoots(metadata_root="/meta", matrix_root="/matrix"),
            provenance=ProvenanceSpec(source_path="/data/test.h5ad", schema="/schema.yaml"),
        )
        d = manifest.to_dict()
        # tokenizer_path is not produced by the manifest in Phase 3 (tokenizer-free)
        assert "tokenizer_path" not in d
