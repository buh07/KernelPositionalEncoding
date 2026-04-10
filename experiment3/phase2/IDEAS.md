# Experiment 3 Phase 2+: Future Experiment Ideas

Evidence snapshot: **2026-04-02** (post-Stage 2 partial completion).

This document collects potential follow-on experiments that go beyond the current Phase 2 protocol. Each entry records the motivating idea, what it would test, and how current Phase 2 findings affect its priority or design.

---

## Idea 1: Word-Level Retokenization and Retraining

**Motivating question:** Is the boundary detection signal (T5b) a deep property of how transformers learn positional structure, or is it an artifact of subword tokenization creating boundaries for the model to detect in the first place?

**Experiment sketch:**
1. Retokenize training data with a strict word-level tokenizer (whitespace-split, no BPE merges). Each token is a complete word.
2. Retrain a small model (1B-scale) from scratch with this tokenizer, matched architecture to Llama/OLMo.
3. Run the full T5b boundary detection protocol. Under word-level tokenization, there are no subword boundaries — only word boundaries exist, and every token *is* a word.
4. Profile SI head R² and compare the R² distribution to the BPE-tokenized baseline.

**What it tests:**
- If high-SI heads still emerge and attend to word boundaries under word-level tokenization, the boundary mechanism is a genuine positional computation, not a response to BPE fragmentation.
- If SI structure collapses or boundary detection disappears, the mechanism is tokenizer-dependent — the model learns to service BPE's segmentation decisions rather than performing a fundamental positional operation.

**Connection to Phase 2 findings:**
- 3P2-B showed the boundary signal is **non-trivial** (d=1.085 Llama, d=0.543 OLMo; no prefix-following artifact). This rules out the simplest artifact (heads just following the space-prefix feature), but does not rule out the deeper question of whether boundaries only matter *because* BPE creates them.
- A word-level retokenization experiment would be a stronger test of E9 (tokenizer/corpus-specific artifact) than the 3P2-I invariance checks, which only vary corpus, not tokenization scheme.
- If the 3P2-B non-triviality result is robust, this experiment asks: non-trivial *given BPE*, but is it non-trivial *in general*?

**Priority:** High. This is the most direct way to resolve the boundary-mechanism interpretation. The retraining cost is the main barrier.

---

## Idea 2: Cross-Lingual Tokenization

**Motivating question:** Does the boundary detection mechanism generalize across languages with different morphological structure and tokenization behavior?

**Experiment sketch:**
1. Run the existing T5b protocol on non-English text using the same pretrained models (Llama and OLMo both have multilingual capacity to varying degrees).
2. Target languages with contrasting properties:
   - **Agglutinative** (e.g., Turkish, Finnish): words are long with many morpheme boundaries; BPE fragments heavily within words.
   - **Isolating** (e.g., Mandarin Chinese): minimal morphological structure; tokenization operates at character/subcharacter level with different boundary semantics.
   - **Fusional** (e.g., Russian, German): moderate morphological complexity with different compounding and case-marking patterns.
3. Compare boundary attention profiles of high-SI heads across languages.
4. Test whether the R²-boundary correlation (T5b Approach C) holds cross-lingually.

**What it tests:**
- If the boundary mechanism is language-universal, high-SI heads should show similar boundary attention patterns regardless of the morphological structure being tokenized.
- If it is language/tokenizer-specific, boundary effects should scale with how much the tokenizer fragments words in that language (strongest in agglutinative languages, weakest in isolating languages).
- This also probes whether E5 (boundary-primary) or E1 (general positional infrastructure) better explains the mechanism: a universal boundary signal favors E5; a signal that tracks tokenizer fragmentation rate favors E9/E1.

**Connection to Phase 2 findings:**
- 3P2-B confirmed non-trivial boundary detection in English. Cross-lingual data would test whether this is a general property or an English/BPE-specific one.
- 3P2-I (Stage 3, not yet executed) plans corpus variation but only within English. This idea extends the invariance question to a much harder test.
- The OLMo T5b Approach B failure (B=N) already hints at fragility. Cross-lingual testing could reveal whether the mechanism is robust in some languages and fragile in others, or uniformly fragile outside English Wikipedia.

**Priority:** Medium. Can be done with existing pretrained models (no retraining), but multilingual evaluation data and tokenization analysis require setup work. Lower barrier than Idea 1.

---

## Idea 3: N-Gram Tokenization

**Motivating question:** BPE tokenizers learn a linguistically informed vocabulary. What happens to SI structure under a tokenizer that has no linguistic prior — pure fixed-length n-gram chunking?

**Experiment sketch:**
1. Retokenize data with fixed character n-gram tokenizers (e.g., bigram, trigram, 4-gram) — no learned merges, no linguistic segmentation.
2. Retrain a matched-architecture small model on each tokenization.
3. Profile SI heads: do high-R² heads still emerge? If so, what do they attend to?
4. Run T5b: under n-gram tokenization, "word boundaries" are arbitrary chunk boundaries. If boundary detection persists, it is a generic segmentation response; if it disappears, it was tracking linguistically meaningful boundaries.

**What it tests:**
- Disentangles whether SI heads learn to detect *linguistically meaningful* boundaries or *any tokenizer-induced* boundaries.
- Under n-gram tokenization, token boundaries have no consistent relationship to word or morpheme boundaries. If boundary detection effects persist, this is evidence for a low-level chunking mechanism (E1-consistent); if they disappear, it is evidence for a linguistically grounded computation (E5-consistent).
- Also tests whether high R² values depend on the variable token lengths that BPE produces (BPE creates natural rhythmic structure in token sequences; fixed n-grams do not).

**Connection to Phase 2 findings:**
- Complements Idea 1 (word-level) by going in the opposite direction: instead of removing subword boundaries, n-gram tokenization makes *all* boundaries arbitrary.
- 3P2-B's space-prefix controls removed one surface feature; n-gram tokenization removes the entire linguistic segmentation prior.
- The 3P2-E reconciliation (T5 vs T5b) showed boundary detection is the primary mechanism, not subword assembly. N-gram tokenization would test whether "boundary" means "linguistically meaningful segmentation point" or just "token edge."

**Priority:** Medium-low. Requires retraining, and the interpretation is less clean than Idea 1 (word-level) because n-gram tokenization degrades model quality substantially, confounding mechanism-level conclusions with capability-level effects.

---

## Idea 4: Structural Ambiguity and Positional Invariance in Semantic Parsing

**Motivating question:** If high-SI heads provide positional infrastructure for parsing, do they participate differently when sentence structure is ambiguous — where the same surface string supports multiple syntactic trees?

**Experiment sketch:**
1. Construct a stimulus set of structurally ambiguous sentences where the parse tree determines meaning:
   - PP-attachment: "I saw the man with the telescope" (instrument vs. possession)
   - Relative clause attachment: "The daughter of the colonel who was on the balcony" (daughter vs. colonel)
   - Coordination: "old men and women" (old modifies both vs. only men)
2. For each ambiguous sentence, construct:
   - A disambiguation context that forces one parse (e.g., prior sentence establishing who owns the telescope)
   - A probe question whose correct answer depends on the parse
3. Run the model under three conditions:
   - (a) Intact model: measure which parse the model prefers (next-token probabilities or probe question accuracy)
   - (b) High-SI heads ablated: does the model shift parse preference? Become less consistent?
   - (c) Low-SI heads ablated: same measurements
4. Key analysis: in a syntactic tree, some structural relationships are shift-invariant (local adjacency, e.g., determiner-noun) and some are absolute/hierarchical (long-range dependencies, e.g., subject-verb agreement across a relative clause). Measure whether high-SI ablation selectively disrupts the local/relative relationships while leaving absolute/hierarchical ones intact, or vice versa.

**What it tests:**
- Whether high-SI heads contribute to syntactic structure building, not just boundary detection.
- The **invariant vs. absolute decomposition** of parse trees: if parts of the tree that correspond to local/relative positioning are disrupted by SI ablation while hierarchical parts are not, this supports a specific functional role for the SI kernel in syntax.
- If SI ablation makes the model *less consistent* in its parse (higher entropy over interpretations without shifting the mode), this supports E1 (general infrastructure reducing uncertainty). If it shifts the parse toward one interpretation, this supports a more specialized role.

**Connection to Phase 2 findings:**
- T1 showed no selective impairment on positional tasks under SI ablation — but those tasks (retrieval, local_key_match) are all unambiguous. Structural ambiguity is a regime where positional infrastructure might matter more, connecting to E10 (context-conditional specialization).
- 3P2-J (not yet executed) plans to test conditional specialization in "hard" regimes. Structural ambiguity is a natural hard regime that 3P2-J's pre-registered definitions (long-span, high-uncertainty, boundary-dense) may not capture.
- The E5 (boundary-primary) interpretation predicts SI heads should not matter much for high-level parse ambiguity (boundaries are low-level). If they do matter, the mechanism is richer than boundary detection.
- The 3P2-A multi-task battery includes a syntactic task (subject-verb agreement) but not ambiguity resolution. This idea fills that gap.

**Priority:** High. This is a qualitatively different test from anything in the current Phase 2 protocol. It directly probes whether the positional kernel participates in higher-level structure, not just token-level boundary marking. The stimulus construction is the main challenge (needs careful norming to ensure genuine ambiguity).

---

## Idea 5: Superposition Within Frequency Bands

**Motivating question:** Theory 9 showed that high-frequency RoPE components correlate with the feeder effect (rho_hf up to +0.438). Are multiple distinct positional computations superposed within the same frequency band, or does each band carry a single function?

**Experiment sketch:**
1. For high-SI heads, decompose the attention logit matrix into per-RoPE-pair contributions (using the existing `rope_freq_per_head.py` infrastructure).
2. Within the high-frequency band (top 25% of pairs by frequency), apply sparse dictionary learning or NMF to the per-pair logit contributions across many sequences.
3. Look for distinct learned components: do different HF pairs specialize for different tasks (boundary detection vs. previous-token attention vs. local copy), or do they all carry the same signal?
4. Validate by selective pair attenuation: attenuate individual HF pairs and measure which downstream tasks are disrupted.

**What it tests:**
- Whether the frequency band is a single functional channel or a multiplexed channel carrying superposed signals.
- If superposition is found: the positional kernel is richer than a single g(Delta) function suggests — different circuits read different spectral components of the same kernel.
- If no superposition (single function per band): the kernel is functionally simpler, and the high-frequency band is a dedicated local-processing channel.
- Connects to the broader mechanistic interpretability question of whether attention heads use frequency bands the way neural populations use mixed selectivity.

**Connection to Phase 2 findings:**
- T9 is descriptive (correlational) — this would make it causal by intervening on individual frequency components.
- Experiment 2's pair-level profiling showed strong pair-specific effects in Llama (pair0 vs pair56 significant) but near-null in OLMo. Superposition analysis could explain why: if OLMo superimposes more functions per pair, per-pair interventions produce weaker effects because removing one pair disrupts multiple functions simultaneously (partial compensation across superposed codes).
- 3P2-G dose-response found OLMo has early selective slopes but Llama does not. If OLMo uses more superposition, graded attenuation would reveal selective effects that binary ablation misses (consistent with E7 saturation in OLMo but not Llama).
- The 3P2-F proxy decomposition showed R² is retained after controlling for head geometry. Frequency-band superposition analysis could identify *what* R² is actually tracking at the sub-head level.

**Priority:** Medium. Analytically interesting and builds on existing infrastructure. The dictionary learning step requires methodological care to avoid overfitting, and validation via pair-level attenuation is GPU-intensive.

---

## Idea 6: Shift-Invariant Channels for Mathematical Reasoning

**Motivating question:** If mathematical operations (counting, arithmetic, sequence continuation) depend on precise relative positioning, can performance be improved by architecturally restricting computation to shift-invariant channels?

**Experiment sketch:**
1. Design a suite of math tasks that depend on positional structure:
   - Counting: "How many items in: A B C D E?" (answer depends on relative token positions)
   - Arithmetic with carries: multi-digit addition where digit alignment is positional
   - Sequence continuation: "2, 4, 8, 16, ?" (pattern recognition over relative offsets)
   - Modular arithmetic: operations where shift-invariance maps directly to the algebraic structure
2. **Intervention approach (no retraining):** Amplify high-SI head contributions (scale o_proj by 1.5x or 2x) while attenuating low-SI heads (scale by 0.5x or 0x) during math task evaluation. This "channels" computation through the SI pathway.
3. **Architectural approach (retraining):** Train a small model variant where a subset of heads are constrained to shift-invariant attention (attention logits forced to be Toeplitz) while remaining heads are unconstrained. Compare math performance to an unconstrained baseline.
4. Measure whether channeling through SI pathways improves or degrades math accuracy.

**What it tests:**
- Whether the shift-invariant kernel is *useful* for math reasoning or merely a general-purpose positional primitive that math tasks happen to consume.
- The intervention approach tests whether existing SI heads already carry math-relevant signal. If amplifying them helps, math circuits are already wired through SI heads.
- The architectural approach tests whether *constraining* heads to be shift-invariant is a useful inductive bias for math. If the Toeplitz-constrained model outperforms on math while underperforming on general language, SI is a math-specific advantage, not a general one.
- This connects to the E1 (general infrastructure) vs. E10 (context-conditional specialization) debate: if SI amplification selectively helps math, E10 gains support; if it helps everything equally, E1 is reinforced.

**Connection to Phase 2 findings:**
- T1 showed no selective impairment from high-SI ablation on positional tasks. But T1's tasks (retrieval, local_key_match) are not mathematical. Math reasoning is a qualitatively different kind of positional demand.
- T8 showed the kernel is functionally important (removing it increases loss). Math tasks with explicit positional alignment requirements (digit-by-digit addition) are the strongest case for the kernel being load-bearing.
- 3P2-G dose-response infrastructure already exists for graded attenuation. The intervention approach (amplify SI, attenuate non-SI) is a natural extension of dose-response methodology applied to math-specific evaluation.
- 3P2-A plans a multi-task battery but focuses on language tasks. Math tasks would extend the battery into a domain where positional structure is more explicitly functional.

**Priority:** Medium. The intervention approach (no retraining) is low-cost and could produce a clean result. The architectural approach (Toeplitz-constrained heads) is higher-cost but would be a novel architectural contribution if it works. Start with the intervention approach as a feasibility check.

---

## Summary Table

| Idea | Core question | Tests explanation(s) | Requires retraining? | Affected by Phase 2 findings |
|---|---|---|---|---|
| 1. Word-level retokenization | Is boundary detection BPE-dependent? | E5, E9 | Yes | 3P2-B passed (non-trivial), but deeper tokenizer dependence untested |
| 2. Cross-lingual tokenization | Does boundary mechanism generalize across languages? | E5, E9, E1 | No | 3P2-B passed for English; OLMo Approach B fragility hints at cross-lingual variation |
| 3. N-gram tokenization | Do SI heads need linguistically meaningful boundaries? | E1, E5, E9 | Yes | 3P2-E reconciliation points to boundary mechanism; n-grams test if "boundary" means linguistic or arbitrary |
| 4. Structural ambiguity | Do SI heads participate in parse structure, not just boundaries? | E1, E5, E10 | No | T1 null may reflect easy tasks; ambiguity is a harder regime (connects to 3P2-J) |
| 5. Frequency band superposition | Is the kernel multiplexed or single-function? | E6, E7 (indirectly) | No | T9 descriptive, Exp2 pair-level Llama/OLMo divergence, 3P2-G dose-response split |
| 6. Math via SI channels | Can SI structure improve math reasoning? | E1, E10 | Intervention: No; Architecture: Yes | T8 shows kernel is functional; 3P2-G dose-response infra reusable |

## Dependency and Sequencing Notes

- **Ideas 1, 3** both require retraining and can share infrastructure. If retraining is planned, run both tokenizer variants (word-level + n-gram) simultaneously.
- **Idea 2** can run immediately on existing pretrained models. It is the lowest-barrier extension of 3P2-I and could be folded into Stage 3 if 3P2-B's gate is passed (which it has been).
- **Idea 4** should wait for 3P2-A and 3P2-J results. If 3P2-A shows uniform task degradation (supporting E1) and 3P2-J finds no conditional specialization, structural ambiguity becomes the next natural test of whether specialization exists in harder regimes.
- **Idea 5** can proceed independently. It builds on existing frequency-decomposition infrastructure and T9 artifacts.
- **Idea 6** intervention approach can run immediately using 3P2-G dose-response code with a math task battery. The architectural approach is a longer-term project.
