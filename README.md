# Measuring Definition Sensitivity in LLM Annotation of Social Science Concepts

This repository contains the code, the analysis tables and the prompt registry. The paper:

```bibtex
@misc{mamaev2026definition,
  title         = {Measuring Definition Sensitivity in {LLM} Annotation of Social Science Concepts},
  author        = {Mamaev, Bogdan and Wang, Qizhou and Vergani, Matteo and Leckie, Christopher},
  year          = {2026},
  eprint        = {2611.13339},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2611.13339},
  url           = {https://arxiv.org/abs/2611.13339},
  note          = {Accepted at the NeurIPS 2026 Workshop TAE (Trust-AI-Eval): Can We Trust AI Evaluation?}
}
```

In the paper, we measure how much the labels of a large language model (LLM) change when only the definition in the
prompt changes. We test six concepts: protest, assassination, terrorism, hate speech, climate contrarianism,
and fake news. We quote competing definitions from published codebooks and studies for each concept. Three models
label each text INCLUDE, EXCLUDE or INDETERMINATE with each definition, and INDETERMINATE means that the text does not
state what the definition requires. We used Claude Sonnet 5 (Sonnet) and two open-weight models, Gemma 4 31B
(Gemma) and Qwen 3.6 27B (Qwen).

The disagreement rate is the share of texts to which a model assigns different labels to two definitions.
Definition sensitivity is the mean disagreement rate over the pairs of competing definitions of a concept.
We introduced paraphrase floor – the mean disagreement rate in the pairs of wordings of the same definition.

## Contents

| Path | What it contains |
|---|---|
| `registry/prompt-registry.json` | One record for each of the 39 prompts |
| `tables/` | The analysis tables |
| `code/` | The code for the annotation, the comprehension check, the analysis and the verification script |

Each prompt pairs one wording of a definition with a query, which is the same for all prompts of a concept.
Each record of the prompt registry contains the IDs of the concept (`construct`), the definition (`def_id`), and the
wording (`impl_id`). It also contains the hash of the prompt (`instrument_sha`) and an estimated token count
(`n_tokens_est`). The last two fields are the version of the prompt set (`suite_version`) and the path of the prompt
file (`file`).

## This repository does not contain

- **The texts.** The news-derived texts are licensed; we cannot redistribute them (refer to Section 3.1 of the paper). We drew the texts for hate speech, climate contrarianism and
  fake news from public datasets (Borkan et al. 2019, Diggelmann et al. 2020, Wang 2017). The script
  `code/sample_expansion.py` retrieves them again from these datasets.
- **The prompt files.**  We quoted the definitions in the prompts from
  the codebooks and studies in the table below (Section 3.2 for more information).
- **The results files.** The labels that the models assigned are not part of this release.

| Concept | `def_id` | Source of the definition |
|---|---|---|
| Protest | `acled` | Raleigh et al. 2010 and ACLED 2026, the codebook of the Armed Conflict Location and Event Dataset (ACLED) |
| Protest | `ccc_2021`, `ccc_2024` | Crowd Counting Consortium (CCC) 2024, the 2021 and the 2024 definitions |
| Protest | `cnts_agd` | Banks and Wilson 2024, the Cross-National Time-Series Data Archive (CNTS) |
| Protest | `mm` | Clark and Regan 2016, the Mass Mobilization Protest Data (MMPD) |
| Assassination | `gtd` | START 2021, the codebook of the Global Terrorism Database (GTD) of the National Consortium for the Study of Terrorism and Responses to Terrorism (START) |
| Assassination | `gtd_success` | START 2021, the GTD definition with successful attacks only |
| Assassination | `cnts` | Banks and Wilson 2024 (CNTS) |
| Assassination | `targeted_killing` | International Committee of the Red Cross 2026 |
| Terrorism | `gtd_crit` | START 2021 (GTD) |
| Terrorism | `schmid` | Schmid 2012 |
| Terrorism | `enders_sandler` | Enders 2007 |
| Terrorism | `weinberg` | Weinberg et al. 2004 |
| Hate speech | `davidson` | Davidson et al. 2017 |
| Hate speech | `waseem_hovy` | Waseem and Hovy 2016 |
| Hate speech | `founta` | Founta et al. 2018 |
| Hate speech | `degibert` | de Gibert et al. 2018 |
| Climate contrarianism | `cards` | Coan et al. 2021 |
| Climate contrarianism | `cook` | Cook et al. 2013 |
| Climate contrarianism | `capstick` | Capstick and Pidgeon 2014 |
| Climate contrarianism | `rahmstorf` | Rahmstorf 2004 |
| Fake news | `lazer` | Lazer et al. 2018 |
| Fake news | `allcott` | Allcott and Gentzkow 2017 |
| Fake news | `gelfert` | Gelfert 2018 |
| Fake news | `zhou` | Zhou and Zafarani 2020 |

## Names in the code and the tables

This table lists the names that the code and the tables use in place of the terms of the paper.

| Name in the code and the tables | Term in the paper |
|---|---|
| arm | model |
| construct | concept |
| def_id | definition |
| impl_id | wording. The ID `impl_0` marks the original wording of the definition, and `impl_1` and `impl_2` mark its paraphrases. |
| instrument | prompt |
| unit | text |
| verdict | label |
| transmission | definition sensitivity |
| gate, G2 | comprehension check |
| chunk | batch, a group of at most 50 texts that Sonnet labelled with one prompt |
| frame_sha | the hash of a sample of texts that we pre-registered before the main annotation run |

## Tables

| File | Description |
|---|---|
| `an_manifest.json` | The models, the bootstrap settings, the hashes of the news-derived sample and of the sample from the public datasets, the number of assassination texts that we excluded because the download returned binary files, and the number of analysed texts for each concept |
| `t1_verdict_matrix.csv` | For each model, concept and prompt, the number of missing labels and the share of each label |
| `t2_transmission_floors.csv` | For each model and concept, definition sensitivity, the paraphrase floor and the gap between them with 95% confidence intervals, and their ratio |
| `t3_ccc_contrast.csv` | For each model, the share of protest texts with the same label with CCC 2021 and CCC 2024, and the change in the INCLUDE share, with 95% confidence intervals |
| `t3_ccc_flips__*.csv` | For one model, the number of protest texts with each pair of labels, with the labels for CCC 2021 in the rows and those for CCC 2024 in the columns |
| `t5_nesting.csv` | For each model, the number of assassination texts labelled INCLUDE with GTD with successful attacks only, and the share of these texts that are also labelled INCLUDE with GTD |
| `t6_crosssite.csv` | For Gemma and Qwen and each prompt of terrorism, hate speech, climate contrarianism and fake news, the share of labels that are the same in the main annotation run and in the re-run on a second platform |
| `t7_repeat.csv` | For each repeat batch, the share of Sonnet's labels that stay the same when Sonnet labels the batch again with the same prompt. The paper calls the share that changes the repeat floor. |
| `t8_effort.csv` | For Sonnet, the disagreement rate of each pair of the three wordings of the ACLED definition. With the original wording, Sonnet labelled part of the texts at the default reasoning effort of the harness and the rest at an effort that we set. |
| `t9_floor_by_def.csv` | For each model, the paraphrase floor of each of the two paraphrased protest definitions, ACLED and MMPD, with definition sensitivity for protest and their ratio |
| `t10_fake_batch_boot.csv` | For Sonnet and fake news, definition sensitivity, the paraphrase floor and the gap, with 95% confidence intervals from a bootstrap that resamples the 12 batches |
| `t11_pairs.csv` | For each model and concept, the disagreement rate of each pair of competing definitions |

## How to run the code

The scripts need the texts, the prompt files and the results files, which this repository does not contain, so make sure to obtain them beforehand.

The scripts need Python 3.11 with pandas, NumPy, pyarrow and Matplotlib. The scripts `label.py` and `gate.py` also need
the `openai` Python package, which sends the requests to the vLLM server of each open-weight model. The file
`code/_paths.py` lists the folder of each input, and this table describes these folders.

| Folder | What the scripts expect to find there |
|---|---|
| `data/` | The two frozen samples, `sample.parquet` and `sample_expansion.parquet`, and the list of excluded texts, `corrupt_fetch_ids.csv` |
| `data/raw/expansion/` | The three public datasets that `sample_expansion.py` reads |
| `defs/instruments/` | The prompt files that the `file` field of the registry names |
| `out/labels/` | One results file for each model and prompt |
| `out/labels_second_platform/` | The results files of the re-run on a second platform |
| `out/labels_qwen_first_run/` | Qwen's first run on the other four concepts, which Appendix A of the paper describes |
| `out/comprehension_check/` | The answers of each model in the comprehension check |
| `out/sonnet_repeat/` | Sonnet's repeat batches |
| `out/sonnet_batches/` | The texts of each batch that Sonnet labelled |

For the labels that the paper reports, we served Gemma and Qwen with vLLM 0.19.1 and transformers 5.14.1. The file
`code/annotators.json` records the revision of each open-weight model. For protest and assassination, we used each
open-weight model on one Nvidia A100 graphics processing unit (GPU) with 80 GB of memory. Sonnet ran through an agentic
command-line harness, which is not part of this repository.

To run the code, do these steps from the root of the repository.

1. Draw the texts for hate speech, climate contrarianism and fake news from the public datasets.

   ```bash
   python3 code/sample_expansion.py
   ```

2. In a separate terminal, serve Gemma.

   ```bash
   vllm serve google/gemma-4-31B-it --revision 842da3794eaa0b77d5f08bae87a17459d91ff475 --served-model-name gemma4-31b --port 8003 --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.92 --enable-prefix-caching --limit-mm-per-prompt '{"image":0,"video":0}'
   ```

3. In a separate terminal, serve Qwen.

   ```bash
   vllm serve Qwen/Qwen3.6-27B --revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 --served-model-name qwen36-27b --port 8001 --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.92 --enable-prefix-caching --limit-mm-per-prompt '{"image":0,"video":0}'
   ```

4. Compute the hash of each prompt file and write the registry.

   ```bash
   python3 code/label.py instruments
   ```

5. Label the texts with Gemma and Qwen.

   ```bash
   python3 code/label.py run
   ```

6. Run the comprehension check with Gemma.

   ```bash
   python3 code/gate.py gemma4_31b
   ```

7. Run the comprehension check with Qwen.

   ```bash
   python3 code/gate.py qwen36_27b
   ```

8. Compute definition sensitivity, the paraphrase floor and tables t1, t2, t3 and t5.

   ```bash
   python3 code/an_transmission.py
   ```

9. Compute table t6 for the re-run on a second platform.

   ```bash
   python3 code/an_crosssite.py
   ```

10. Compute tables t7 and t8 for Sonnet's repeat batches and reasoning effort.

    ```bash
    python3 code/an_repeat.py
    ```

11. Compute tables t9, t10 and t11.

    ```bash
    python3 code/an_reviews.py
    ```

12. Draw the two panels of Figure 2 of the paper in `figs/`.

    ```bash
    python3 code/an_figures.py
    ```

13. Write the numbers and three tables of the paper as LaTeX files in `paper/`.

    ```bash
    python3 code/an_numbers.py
    ```

14. Run the verification script. It runs 250 checks on the results files and the analysis tables. Five of these
    checks recompute reported values from Gemma's labels with code that is separate from the analysis code.

    ```bash
    python3 code/verify_results.py
    ```

## Licences

- The code in `code/` is under the MIT licence (`LICENSE`).
- The files in `registry/` and `tables/` are under the Creative Commons Attribution 4.0 International licence,
  CC BY 4.0 (`LICENSE-DATA.md`).

## Citation

```bibtex
@misc{mamaev2026definition,
  title         = {Measuring Definition Sensitivity in {LLM} Annotation of Social Science Concepts},
  author        = {Mamaev, Bogdan and Wang, Qizhou and Vergani, Matteo and Leckie, Christopher},
  year          = {2026},
  eprint        = {2611.13339},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2611.13339},
  url           = {https://arxiv.org/abs/2611.13339},
  note          = {Accepted at the NeurIPS 2026 Workshop TAE (Trust-AI-Eval): Can We Trust AI Evaluation?}
}
```
