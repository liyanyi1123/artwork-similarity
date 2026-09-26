# Detailed English Paper Outline

## Working Title

**Joint CLIP-ViT Thresholding for Transformation-Robust Artwork Matching: An Evaluation on ArtBench-10 and WikiArt**

Alternative titles:

1. **Transformation-Robust Artwork Matching with Joint CLIP and ViT Similarity Thresholds**
2. **Threshold Selection for CLIP-ViT Artwork Similarity Verification on ArtBench-10 and WikiArt**
3. **Evaluating Joint Vision-Encoder Thresholds for Near-Duplicate Detection in Art Images**

## Paper Configuration Record

- Paper type: Empirical research article
- Structure: IMRaD with a separate Related Work section
- Field: Computer vision, image retrieval, and computational analysis of art images
- Language: English
- Target length: Approximately 6,000 words, excluding abstract and references
- Primary evaluation metric: F1 score
- Secondary metrics: Accuracy, precision, recall, specificity, false-positive rate, and confusion-matrix counts
- Citation style: To be selected after the target venue is chosen
- Output scope: Detailed outline and evidence map only

## Scope Lock

The manuscript should discuss only the following two 1,000-image evaluation sets:

1. **ArtBench-1K**: 1,000 images sampled from ArtBench-10, with 100 images from each of 10 art styles.
2. **WikiArt-1K**: 1,000 WikiArt images, with 100 images from each of the same 10 art styles.

Do not mention any other exploratory datasets in the main paper. Replace internal directory labels with the publication-facing names `ArtBench-1K` and `WikiArt-1K`.

The paper should compare results from the two evaluation sets, but it should **not claim or attempt to prove cross-dataset generalization**. Use phrases such as "consistency across the two evaluated samples" or "dataset-specific threshold comparison," not "generalization across domains."

## Research Focus

### Central Research Question

How effectively can a joint CLIP-ViT cosine-similarity rule identify transformed versions of an artwork while rejecting pairs of different artworks?

### Sub-questions

- **RQ1:** Which combination of CLIP and ViT similarity thresholds maximizes F1 on ArtBench-1K and WikiArt-1K?
- **RQ2:** How similar are the optimal operating regions and performance profiles of the two evaluated samples?
- **RQ3:** Which image transformations and intensity levels produce the largest reductions in matching recall?
- **RQ4:** What practical trade-off between false positives and false negatives is produced by the joint threshold rule?

### Core Claim Hierarchy

- **Primary claim:** Joint CLIP-ViT thresholding achieves F1 scores above 0.99 on both 1,000-image art evaluation sets.
- **Secondary claim:** The best threshold combinations are close but not identical: CLIP 0.87 with ViT 0.50 for ArtBench-1K, and CLIP 0.88 with ViT 0.48 for WikiArt-1K.
- **Transformation claim:** Most photometric and local-detail transformations are matched almost perfectly, while strong vignette and high rotation are the main sources of false negatives.
- **Methodological claim:** A full two-dimensional threshold sweep reveals the precision-recall trade-off more clearly than selecting a threshold for either encoder in isolation.

## Detailed Outline

### Abstract (200-250 words, not included in the main word count)

**Purpose:** Summarize the problem, method, evaluation design, key numerical findings, and restrained conclusion.

**Content plan:**

- Introduce the need to recognize transformed or near-duplicate artwork images while rejecting different artworks.
- State that normalized image embeddings are extracted from pretrained CLIP ViT-B/32 and ViT-Base/16 models.
- Describe the joint decision rule: a pair is classified as matching only when both CLIP and ViT cosine similarities exceed their respective thresholds.
- State the evaluation scale: two 1,000-image art samples; 31,000 positive original-transform pairs and 499,500 negative original-original pairs per sample.
- State the threshold search: CLIP and ViT thresholds from 0.00 to 1.00 in increments of 0.01, producing 10,201 combinations.
- Report the ArtBench-1K optimum: CLIP 0.87, ViT 0.50, F1 0.994475.
- Report the WikiArt-1K optimum: CLIP 0.88, ViT 0.48, F1 0.992859.
- Note that high-intensity vignette and rotation are the most difficult transformations.
- Conclude that joint thresholding is effective for transformation-based artwork matching, while avoiding broader claims about semantic similarity or cross-dataset generalization.

### 1. Introduction (approximately 750 words)

**Purpose:** Establish the practical and research problem, define the task precisely, and state the paper's contribution.

#### 1.1 Artwork reuse and transformation-based matching

- Explain that digital artworks are frequently resized, cropped, shifted, recolored, sharpened, compressed, or otherwise altered during online circulation and archival processing.
- Distinguish transformation-based identity matching from general semantic image similarity.
- Explain the two error types: missing a transformed copy of the same artwork and falsely matching two different artworks.

#### 1.2 Limitations of a single similarity representation

- Introduce CLIP as a semantically rich image representation and ViT as a complementary visual representation.
- Explain why one threshold on one model may preserve recall but admit visually related false positives, or suppress false positives at the cost of false negatives.
- Motivate a joint rule that requires agreement between both similarity measures.

#### 1.3 Study objective and research questions

- State the central RQ and RQ1-RQ4.
- Define the empirical objective as threshold characterization and error analysis on two art-image samples.
- Explicitly state that the study does not evaluate broad domain generalization.

#### 1.4 Contributions

- A controlled evaluation protocol using two balanced 1,000-image art samples.
- A transformation suite covering geometric, photometric, spatial, and local-detail changes at multiple intensities.
- A complete 101 x 101 CLIP-ViT threshold sweep rather than a small manually selected threshold set.
- A transformation-level error analysis identifying the operations that most strongly reduce recall.
- A reproducible evidence package containing similarity arrays, threshold tables, heatmaps, and scripts.

**Transition to Section 2:** The introduction motivates joint thresholding; the related-work section positions it within representation-based image similarity and artwork analysis.

### 2. Related Work (approximately 900 words)

**Purpose:** Establish the conceptual basis for the method without turning the paper into a broad survey.

#### 2.1 Image similarity and near-duplicate detection

- Review the distinction among exact duplicate detection, perceptual hashing, near-duplicate detection, semantic retrieval, and identity-preserving transformation matching.
- Position the present task as supervised-by-construction binary verification rather than open-ended semantic similarity.

#### 2.2 Vision transformers and CLIP image embeddings

- Summarize cosine similarity over normalized neural embeddings.
- Discuss the different inductive roles of CLIP and a standard pretrained ViT.
- Explain why complementary embedding spaces may benefit a conservative AND fusion rule.

#### 2.3 Robustness to image transformations

- Review prior evaluation practices for crop, shift, rotation, brightness, color, noise, perspective, and local-detail alterations.
- Identify the need for operation-level and intensity-level reporting rather than only aggregate accuracy.

#### 2.4 Computational analysis of art images

- Introduce ArtBench-10 and WikiArt as art-image resources organized by artistic style.
- Use the local review paper by Cetinic and She (2022) for broad AI-and-art context after verifying the exact relevant passages.
- Use other PDFs in `Related_papers/` only after bibliographic and claim-level verification.

#### 2.5 Research gap

- State that relatively little work reports a dense, joint CLIP-ViT threshold landscape for transformation-based artwork matching.
- Emphasize the gap in connecting aggregate performance to specific transformation types and intensities.

**Transition to Section 3:** The identified gap leads to a controlled dual-encoder threshold experiment with a fixed pair-construction protocol.

### 3. Methodology (approximately 1,400 words)

**Purpose:** Make the experiment reproducible and ensure that every reported metric can be traced to a defined pair set and decision rule.

#### 3.1 Experimental design

- Describe a repeated evaluation design applied separately to ArtBench-1K and WikiArt-1K.
- State that the same art-style categories, transformation functions, pretrained models, pair definitions, threshold grid, and metrics are used for both samples.
- Clarify that no model fine-tuning is performed.

#### 3.2 Dataset construction

##### 3.2.1 ArtBench-1K

- 1,000 original images from ArtBench-10.
- Ten styles: art nouveau, baroque, expressionism, impressionism, post-impressionism, realism, renaissance, romanticism, surrealism, and ukiyo-e.
- One hundred images per style.

##### 3.2.2 WikiArt-1K

- 1,000 WikiArt images organized into the same ten styles.
- One hundred images per style.
- Present this as a second evaluation sample, not as evidence of domain generalization.

##### 3.2.3 Dataset controls

- Equalize the number of originals, style counts, transformations per original, and pair counts.
- Explain image validation, supported image formats, RGB conversion, and handling of unreadable files.
- Include a short data-provenance statement and the sampling procedure used for each set.

#### 3.3 Transformation protocol

- State that 32 transformed files are generated per original image from 12 operations.
- Explain that the current evaluation excludes vertical flip (`F2`), leaving 31 evaluated transformed pairs per original.
- Either justify the exclusion of `F2` in the manuscript or include it in a revised analysis; do not state that all 32 transforms were evaluated unless the evaluation is rerun.

Transformation families:

- Horizontal flip (`F1`), one level.
- Vertical flip (`F2`), generated but excluded from the current threshold evaluation.
- Center crop and resize (`C`), low/mid/high.
- Horizontal shift (`S1`), low/mid/high.
- Vertical shift (`S2`), low/mid/high.
- Rotation (`R`), low/mid/high.
- Brightness and contrast adjustment (`B`), low/mid/high.
- Vignette (`V`), low/mid/high.
- Color-temperature shift (`T`), low/mid/high.
- Sharpening (`Sh`), low/mid/high.
- Perspective skew (`K`), low/mid/high.
- Film grain or Gaussian noise (`N`), low/mid/high.

#### 3.4 Representation models and similarity computation

- CLIP model: `openai/clip-vit-base-patch32`.
- ViT model: `google/vit-base-patch16-224`.
- Convert images to RGB and apply each model's native processor.
- Extract CLIP image features and the ViT CLS-token representation.
- L2-normalize embeddings and calculate cosine similarity by dot product.
- State model versions, Transformers/PyTorch versions, hardware, batch sizes, and random seeds in the final manuscript.

#### 3.5 Positive and negative pair construction

- Positive pairs: each original paired with its 31 evaluated transformed versions, yielding 31,000 positive pairs per dataset.
- Negative pairs: every unordered pair of different originals, yielding C(1000, 2) = 499,500 negative pairs per dataset.
- Total: 530,500 labeled pairs per dataset.
- Explain that the negative class is much larger, so accuracy alone is insufficient.
- Clarify that pair labels encode image identity under transformation, not subjective semantic similarity.

#### 3.6 Joint CLIP-ViT decision rule

Define a pair as matching when:

`CLIP_similarity > t_clip AND ViT_similarity > t_vit`

- Explain that CLIP and ViT must both pass their thresholds.
- If the implementation is described as a two-stage pipeline, state that its final mathematical decision is equivalent to the AND rule.
- Define TP, FP, TN, and FN using the positive and negative pair sets.

#### 3.7 Threshold search

- Sweep `t_clip` from 0.00 to 1.00 in steps of 0.01.
- Sweep `t_vit` from 0.00 to 1.00 in steps of 0.01.
- Evaluate 101 x 101 = 10,201 threshold combinations separately for each dataset.
- Select the primary operating point by maximum F1, with accuracy used as a secondary check.
- Report the surrounding top-performing region, not only the single best grid point, because nearby settings have nearly identical F1 values.

#### 3.8 Evaluation metrics

- Precision, recall, F1, accuracy, specificity, and false-positive rate.
- Confusion-matrix counts for transparent interpretation.
- Transformation-level recall by operation and intensity.
- Avoid formal significance claims unless uncertainty estimates are added.

#### 3.9 Reproducibility and data management

- Reference the threshold-sweep scripts, cached embeddings, similarity arrays, result CSVs, and generated figures.
- State that embeddings are cached and threshold combinations are evaluated without repeatedly loading the models.
- Provide a reproducibility appendix listing exact file locations and commands.

**Transition to Section 4:** The methodology defines the search space and metrics; the results first report the global operating points and then localize errors by transformation.

### 4. Results (approximately 1,400 words)

**Purpose:** Present the two datasets cleanly, beginning with aggregate results and then explaining where errors occur.

#### 4.1 Optimal threshold combinations and aggregate performance

Report the following primary table:

| Evaluation set | CLIP threshold | ViT threshold | F1 | Accuracy | Precision | Recall | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ArtBench-1K | 0.87 | 0.50 | 0.994475 | 0.999355 | 0.996084 | 0.992871 | 30,779 | 121 | 499,379 | 221 |
| WikiArt-1K | 0.88 | 0.48 | 0.992859 | 0.999167 | 0.994594 | 0.991129 | 30,725 | 167 | 499,333 | 275 |

Interpretation points:

- Both samples produce F1 above 0.99.
- ArtBench-1K has 46 fewer false positives and 54 fewer false negatives at its own optimum.
- The absolute F1 difference is 0.001616; describe it as a small descriptive difference, not automatically as a statistically meaningful difference.
- Accuracy is very high partly because negative pairs dominate; F1 and the confusion matrix should lead the interpretation.

#### 4.2 Threshold-performance landscapes

- Present the two F1 heatmaps side by side with identical axes and color scales.
- Mark each sample's best point.
- Show that multiple neighboring threshold combinations yield nearly identical F1 values.
- Describe the optimum as a high-performance region rather than implying that the second decimal place is a universal constant.
- Use the top-10 threshold tables in the supplementary material.

#### 4.3 Comparison of the two evaluated samples

- ArtBench-1K optimum: CLIP 0.87 and ViT 0.50.
- WikiArt-1K optimum: CLIP 0.88 and ViT 0.48.
- The CLIP threshold differs by 0.01 and the ViT threshold by 0.02.
- Discuss this only as dataset-specific threshold variation and consistency of the general operating region.
- Do not label this subsection "Cross-dataset generalization."

Optional sensitivity note:

- Applying the WikiArt-1K optimum to ArtBench-1K yields F1 0.993951, compared with 0.994475 at the ArtBench optimum.
- Applying the ArtBench-1K optimum to WikiArt-1K yields F1 0.992661, compared with 0.992859 at the WikiArt optimum.
- These small decreases can support a threshold-sensitivity discussion, but should not be presented as a generalization test.

#### 4.4 Robustness by transformation family

Report operation-level recall at each dataset's own optimal thresholds:

| Operation | ArtBench-1K recall | WikiArt-1K recall | Interpretation |
|---|---:|---:|---|
| Brightness/contrast | 1.000000 | 1.000000 | Fully preserved |
| Noise | 1.000000 | 1.000000 | Fully preserved |
| Sharpening | 1.000000 | 1.000000 | Fully preserved |
| Color temperature | 0.999667 | 0.999333 | Nearly invariant |
| Perspective skew | 0.998667 | 0.997667 | Highly robust |
| Crop/resize | 0.997667 | 0.997000 | Highly robust |
| Vertical shift | 0.996333 | 0.994667 | Small degradation |
| Horizontal shift | 0.994000 | 0.990333 | Moderate degradation at high intensity |
| Horizontal flip | 0.991000 | 0.990000 | Small but consistent loss |
| Rotation | 0.971667 | 0.967333 | One of the hardest operations |
| Vignette | 0.971333 | 0.965333 | One of the hardest operations |

#### 4.5 Intensity-level error concentration

- Show that most errors are concentrated in the high-intensity variants.
- ArtBench-1K high vignette recall: 0.916; high rotation recall: 0.929.
- WikiArt-1K high vignette recall: 0.897; high rotation recall: 0.922.
- Contrast these results with perfect recall for all brightness, noise, and sharpening levels.
- Use a grouped bar chart or dot plot rather than a large raw table in the main text.

#### 4.6 Optional fusion-rule ablation

- Existing ArtBench results at fixed CLIP 0.80 and ViT 0.50 show:
  - AND fusion: F1 0.981412.
  - CLIP only: F1 0.933806.
  - ViT only: F1 0.848584.
  - OR fusion: F1 0.812998.
- Use this as secondary evidence that the AND rule reduces false positives.
- Clearly label it as a fixed-threshold ArtBench ablation rather than a direct comparison of each method at its independently optimized threshold.
- Place it in the supplement if space is limited.

**Transition to Section 5:** The results establish aggregate performance and identify the difficult transformations; the discussion explains what these patterns mean and what they do not demonstrate.

### 5. Discussion (approximately 1,200 words)

**Purpose:** Interpret the findings conservatively and connect the numerical results to the design of artwork-matching systems.

#### 5.1 Why joint thresholding is effective

- Explain that requiring both models to agree suppresses pairs that appear similar in only one representation space.
- Relate the high precision and low false-positive counts to the conservative AND rule.
- Explain the remaining false-negative cost visible in rotation and vignette cases.

#### 5.2 Transformation-specific behavior

- Explain why brightness, noise, sharpening, and color-temperature changes may preserve global layout and content features.
- Discuss why strong vignette removes peripheral visual information and changes the global intensity distribution.
- Discuss why rotation and spatial shifts alter patch alignment and spatial correspondence.
- Treat these explanations as interpretations unless supported by additional attention-map or embedding analyses.

#### 5.3 Interpreting the two threshold optima

- Emphasize that 0.87/0.50 and 0.88/0.48 are nearby operating points.
- Explain that a small shift in the maximizing grid cell can occur within a broad, nearly flat high-F1 region.
- Avoid claiming that the threshold differences by themselves prove model sensitivity or generalization.
- Recommend choosing an operating point based on application costs if false positives and false negatives have unequal consequences.

#### 5.4 Practical implications

- Artwork archives and retrieval systems can use a lenient first representation followed by a stricter joint confirmation rule.
- Thresholds should be calibrated for the intended collection and error costs.
- High-rotation and strong-vignette cases may require augmentation, orientation normalization, or a fallback verification stage.

#### 5.5 Limitations

- Thresholds are selected and reported on the same evaluation pairs; the results characterize the available samples rather than provide a held-out estimate.
- Pair observations are not statistically independent because many pairs share the same original image.
- The negative class is much larger than the positive class.
- Positive labels represent synthetic transformations of the same file, not human judgments of perceptual similarity.
- Vertical flip was generated but excluded, so only 31 transformations per original enter the reported evaluation.
- The threshold grid resolution is 0.01; finer numerical precision is not supported.
- No confidence intervals or image-level bootstrap estimates are currently reported.
- The study compares two art-image samples but does not test broad cross-domain generalization.
- Results apply to the selected pretrained CLIP and ViT checkpoints and may differ for other encoders.

#### 5.6 Future work

- Add image-level or style-stratified bootstrap confidence intervals.
- Evaluate orientation normalization or augmentation for rotation failures.
- Investigate adaptive thresholds conditioned on transformation type or collection characteristics.
- Compare the joint rule with learned fusion, metric learning, and dedicated near-duplicate detectors.
- Extend evaluation to real-world edited copies and human-annotated ambiguous pairs.

**Transition to Section 6:** The discussion narrows the claims to transformation-based artwork verification and leads to a concise statement of the main empirical result.

### 6. Conclusion (approximately 350 words)

**Purpose:** Answer the research questions without repeating the entire results section.

- Restate the task as transformation-based artwork matching.
- Summarize the joint CLIP-ViT threshold method and the two 1,000-image evaluation samples.
- Report both optimal threshold combinations and F1 scores.
- State that both samples show a similar high-performing threshold region, while their exact best grid points differ slightly.
- Identify strong vignette and rotation as the main failure modes.
- Conclude that joint thresholding is a simple and effective baseline for transformed-artwork verification.
- End with the need for held-out calibration, uncertainty estimates, and real-world near-duplicate evaluation before broader deployment claims.

## Planned Figures and Tables

### Main-text figures

1. **Figure 1 - Method overview:** Original images, transformations, CLIP and ViT embedding extraction, cosine similarities, AND decision rule, and threshold sweep.
2. **Figure 2 - Transformation examples:** One artwork shown with representative low/mid/high transformations, especially rotation and vignette.
3. **Figure 3 - F1 threshold landscapes:** ArtBench-1K and WikiArt-1K heatmaps with identical axes and marked optima.
4. **Figure 4 - Recall by transformation:** Grouped bars for the 11 evaluated operation families on both samples.
5. **Figure 5 - Recall by intensity for difficult operations:** Rotation, vignette, and horizontal shift at low/mid/high levels.

### Main-text tables

1. **Table 1 - Dataset and pair construction:** 1,000 originals, 10 styles, 31,000 positives, 499,500 negatives, and 530,500 total pairs for each sample.
2. **Table 2 - Transformation definitions and parameter ranges.**
3. **Table 3 - Optimal thresholds and aggregate metrics.**
4. **Table 4 - Transformation-level recall.**

### Supplementary material

- Table S1: Top 10 threshold combinations for ArtBench-1K.
- Table S2: Top 10 threshold combinations for WikiArt-1K.
- Table S3: Full transformation-intensity recall table.
- Table S4: Fixed-threshold AND/OR/single-model ablation.
- Figure S1: Precision or false-positive heatmaps.
- Reproducibility appendix: scripts, model checkpoints, software versions, hardware, seeds, and result-file mapping.

## Evidence Map

| Planned claim or section | Local evidence | Status |
|---|---|---|
| ArtBench-1K source and 100-per-style construction | `Datasets5/ArtBench-10.csv`; `Datasets5/download_100_per_style.py` | Available |
| WikiArt-1K source records | `Datasets6/sources.csv`; `Datasets6/download_wikiart_1000.py` | Available; describe sampling transparently |
| Transformation definitions and parameters | `generate_datasets5_transforms.py`; `exp4/generate_datasets6_transforms.py` | Available |
| CLIP and ViT checkpoint names and embedding extraction | `run_datasets5_threshold.py`; `exp4/run_datasets6_sweep.py` | Available |
| ArtBench-1K optimum and confusion matrix | `exp4/threshold_eval/full_sweep/best_results.json` | Available |
| WikiArt-1K optimum and confusion matrix | `exp4/datasets6_sweep/best_results.json` | Available |
| Complete ArtBench threshold surface | `exp4/threshold_eval/full_sweep/threshold_sweep_summary.csv` | Available |
| Complete WikiArt threshold surface | `exp4/datasets6_sweep/threshold_sweep_summary.csv` | Available |
| ArtBench F1 heatmap | `exp4/threshold_eval/full_sweep/f1_heatmap.png` | Available |
| WikiArt F1 heatmap | `exp4/datasets6_sweep/f1_heatmap.png` | Available |
| Transformation-level recall | CLIP and ViT positive-similarity arrays plus transform paths in the two embedding directories | Derived; save a reproducible summary before submission |
| Fixed-threshold fusion ablation | `exp4/logic_compare/summary_comparison.json` | Available; ArtBench only |
| AI-and-art background literature | `Related_papers/` | Available; each claim and citation must be verified |

## Word Count Summary

| Section | Target words |
|---|---:|
| Introduction | 750 |
| Related Work | 900 |
| Methodology | 1,400 |
| Results | 1,400 |
| Discussion | 1,200 |
| Conclusion | 350 |
| **Total main text** | **6,000** |
| Abstract | 200-250, excluded from total |

## Terminology and Claim Controls

- Use **ArtBench-1K** and **WikiArt-1K**, not `Datasets5` and `Datasets6`.
- Use **transformation-based artwork matching**, **near-duplicate verification**, or **identity-preserving image matching**.
- Do not use broad claims about human perception unless supported by human-subject evidence.
- Do not claim cross-dataset generalization.
- Do not describe the exact best threshold as universal; describe a high-performing threshold region.
- Do not claim that a small threshold change "proves" sensitivity to data distribution.
- Do not report all 32 transformations as evaluated while `F2` remains excluded.
- Lead with F1, precision, recall, and confusion-matrix counts; treat accuracy as secondary because of class imbalance.
- Keep fixed-threshold ablation results separate from independently optimized model comparisons.

## Decisions Required Before Full Drafting

1. Select the target journal or conference and its word limit and citation style.
2. Decide whether to justify the exclusion of vertical flip or rerun the evaluation with all 32 transforms.
3. Decide whether the paper is framed strictly as threshold characterization or whether a held-out calibration/test split will be added.
4. Verify the literature in `Related_papers/` and build a claim-level citation matrix.
5. Generate and save the transformation-level recall table and figures from the cached similarity arrays.

criteria_binding_unavailable
