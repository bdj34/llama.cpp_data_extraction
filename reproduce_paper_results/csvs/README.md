# Data Dictionary & Notes

## prevalence_CDW_AND_MVP

- **Description:**  
  Numbers of reports where the “Microscopic Description” was run (and the number which are positive) for Gemma-2-9B-SPPO and Llama-3-8B, as well as the counts for when “both” are positive or “either” are positive.
- **Columns:**
  - **Method:** Combines the task, cohort, and source for a unique entry for each row.
  - **Task:** The diagnostic task.
  - **Cohort:** Either IBD patient population or non-IBD.
  - **Source:** Data source, either MVP or CDW.
  - **N_llama3_8B / N_gemma2_9B_SPPO:** Number of reports run for Llama-3 and Gemma-2, respectively.
  - **N_overlap:** Number of reports where both models are run.  
    *Note: This is not the number of reports validated, but the larger set in which the models were run. The validation sets are a subset of this larger set of N_overlap.*
  - **N_pos...:** Number positive for each model, including:
    - **both:** Both Llama-3 and Gemma-2 respond “Yes”
    - **either:** Either Llama-3 or Gemma-2 responds “Yes”
  - **Mean/Median Input Chars:** Mean and median number of input characters for the Microscopic Description sections seen by the models.

---

## validation_CDW_AND_MVP

- **Description:**  
  De-identified validation results for all tasks, cohorts, and sources for all models.
- **Columns:**
  - **First six columns:** Model binary outputs (1/0 or TRUE/FALSE).  
    - 1/TRUE = “Yes”; 0/FALSE = “No”; NA = model not run for this report.
    - “_full” suffix: Input was the full pathology report, not just the Microscopic description.
  - **BDJ, AD, HE, LJ:** Reviewers and their validation results (0 = no diagnosis, 1 = diagnosis present).
    - **HE:** Reviewed cases where BDJ and AD disagreed.
    - **LJ:** Reviewed cases where HE was unsure.
    - **NA:** Reviewer did not review that report.
  - **consensusValidation:** Consensus of reviewers:
    - If BDJ and AD agree, their response is the consensus.
    - If not, HE resolves the dispute.
    - If HE is unsure, LJ determines the consensus.
  - **BDJ_full:** Validation result for the full pathology report (see Supplementary Figure S6).
    - BDJ reviewed full reports if either model gave a different answer on the full report vs. the Microscopic section.
    - Otherwise, BDJ_full = consensusValidation.
  - **task:** Diagnostic task evaluated.
  - **cohort:** Patient population (IBD vs. non-IBD).
  - **nchar / ncharFull:** Number of characters (using R’s `nchar()`), for Microscopic description section and full report, respectively.
  - **source:** Data source (MVP or CDW).

---

## all_results

- **Description:**  
  Main results shown in the paper. Converted to `.xlsx` for creating the main and supplementary tables (S3-S5, S8). Produced by the R script `make_csv_from_raw_data.R`.
- **Column Notes:**
  - **w:** Prevalence
  - **lb (lb_boot):** Lower bound of 95% confidence interval (bootstrapping)
  - **ub (ub_boot):** Upper bound of 95% confidence interval (bootstrapping)
  - **upperBound_analytical / lowerBound_analytical:** Binomial 95% confidence interval bounds for NPV and PPV. The wider of analytical or bootstrapping is used (bootstrapping can be too narrow when NPV/PPV = 1).
  - **MCC:** Matthew’s correlation coefficient
  - **N_<>_<>_<>:** Number of reports where the listed conditions are true.
  - **BDJ_<pos/neg>_AD_<pos/neg>:** 2x2 confusion matrix entries for BDJ and AD.
  - **conditionalModel:** Model used for conditioning/stratifying validation.  
    - Typically Llama-3-8B; for IBD-CRC in MVP, “either” (reviewed all reports where either model was positive, and N where both were negative).
- **For more details:**  
  See the R script (`make_csv_from_raw_data.R`) or contact [brian.d.johnson97@gmail.com](mailto:brian.d.johnson97@gmail.com) if anything is unclear.
