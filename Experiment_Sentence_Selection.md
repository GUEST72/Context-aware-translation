# Experiment Log: Capturing the Most Useful Sentence per Term

Date: 2026-03-20
Branch: feature/phase1-context-variants
Author perspective: iterative engineering experiment focused on translation usefulness

## 1) Why I started this experiment

The existing approach stored example sentences, but it was mostly a first-match strategy.
For translation support, that is often weak because:
- first match can be non-definitional
- first match can be noisy or too generic
- multiple paragraphs can share the same metadata, causing wrong sentence retrieval

Goal of this experiment:
Select the sentence that is most useful for translation decisions, not just the first sentence that contains the term.

## 2) Idea sources and papers reviewed

I used practical ideas from summarization and retrieval literature:

1. TextRank (Mihalcea and Tarau, EMNLP 2004)
- Why relevant: unsupervised sentence salience and graph-based ranking.
- How used here: inspired salience-style scoring, but I used a lighter weighted scoring function for this codebase.

2. MMR (Carbonell and Goldstein, 1998)
- Why relevant: balances relevance and diversity.
- How used here: after selecting the best sentence, the second sentence is chosen to add new context instead of repeating the same wording.

3. Sentence-BERT (Reimers and Gurevych, 2019)
- Why relevant: efficient semantic sentence matching.
- How used here: inspired the semantic similarity component. In this implementation I used TF-IDF cosine for lightweight integration, with the design ready to swap to SBERT later.

4. MatchSum (Zhong et al., ACL 2020)
- Why relevant: extraction framed as semantic matching.
- How used here: treated "term + translation intent" as a query and candidate sentences as matching targets.

## 3) Experimental plan

I ran this as three implementation iterations.

### Iteration A: Baseline from Phase 1
- Method: first sentence in each matching paragraph that contains term variant.
- Result: better than no context, but frequent misses in sentence quality and relevance.

### Iteration B: Hybrid weighted scorer
Implemented in src/context_extractor.py:

For each candidate sentence, compute:
- exact_match_score
- semantic_similarity_score (TF-IDF cosine)
- definitional_score (patterns: "is a", "refers to", "defined as", "means")
- domain_salience_score (content-word density)
- sentence_quality_score (length window)

Final score:
- 0.30 * exact
- 0.25 * semantic
- 0.20 * definitional
- 0.15 * salience
- 0.10 * quality

Then select with MMR for diversity.

### Iteration C: Fixing a retrieval bug discovered during experiment
Observed issue:
- Some terms selected wrong sentences even with better scoring.
- Root cause: source location metadata (chapter/heading/page) was not unique.
- Multiple paragraphs shared the same metadata, so lookup could return the wrong paragraph.

Fix:
- Added segment_index to source_locations in src/term_extractor.py.
- Context lookup in src/context_extractor.py now resolves by segment_index first.

This was the largest quality improvement in practice.

## 4) What I changed in code

1. src/context_extractor.py
- Replaced first-match strategy with hybrid ranking.
- Added MMR-based selection.
- Added structured output:
  - example_sentences
  - primary_example_sentence
  - supporting_example_sentences
  - example_score_breakdown

2. src/term_extractor.py
- Added segment_index to each source location so sentence retrieval is deterministic.

3. src/pipeline.py
- Integrated structured output from context extractor into terminology entries.

4. src/terminology_memory.py
- Extended TerminologyEntry with:
  - primary_example_sentence
  - supporting_example_sentences
  - example_score_breakdown
- Kept backward compatibility with example_sentences.

5. data/terminology_memory.json
- Regenerated with the new fields for all terms.

## 5) Concrete findings from this run

Input:
- data/sample_book.json

Runtime:
- Pipeline completes in around 1.0 to 1.2 seconds on this dataset with no embeddings/translation.

Output quality findings:

1. Better explanatory sentence selection
- Terms with definitional contexts are now promoted because definitional_score is explicit.

2. Better diversity between first and second examples
- Supporting sentence is less likely to duplicate the first sentence due to MMR.

3. Bug-driven improvement (most important practical gain)
- Adding segment_index fixed wrong-paragraph retrieval.
- Example: "application" now retrieves true application-containing sentences instead of unrelated nearby content.

4. Better transparency
- Each term now includes example_score_breakdown, so sentence choice is explainable and debuggable.

## 6) Limitations observed

1. Semantic score currently uses TF-IDF cosine, not SBERT embeddings.
- Works reasonably on this small corpus.
- May underperform on broader technical corpora with paraphrase-heavy language.

2. Definitional regex is heuristic.
- High precision for simple patterns, limited recall for complex definitions.

3. MMR diversity is sentence-level only.
- It does not yet model discourse-level novelty across chapters.

## 7) Recommended next experiment

1. Replace TF-IDF semantic component with SBERT (optional path)
- Keep same weighted framework.
- Only swap semantic_similarity_score backend.

2. Add weak-supervision evaluation set
- Label 100 terms with best sentence by human judgment.
- Compare:
  - baseline first-match
  - hybrid TF-IDF
  - hybrid SBERT

3. Tune weights with grid search over that labeled set.

## 8) Final conclusion

The experiment succeeded.

Most important outcome:
- The system now selects translation-helpful sentences with explicit scoring logic and diversity control.
- The segment_index fix made sentence retrieval reliable, which was critical for overall quality.

Practical impact for translators:
- More explanatory primary examples
- Better secondary context
- Explainable score breakdown for trust and manual review
