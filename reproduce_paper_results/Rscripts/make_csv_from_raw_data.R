# Take VINCI outputs and make a single csv
library(tidyr)
library(dplyr)
library(gt)

rm(list=ls())

# Set the number of bootstraps to perform. 10k is overkill but that's what we did
n_bootstrap <- 10000

# source the function for computing the metrics
source("~/VA_IBD/llm_paper_tables_figures_sources_2025_04_22/resubmission_2025_06/Rscripts/fn_compute_stats.R")

#### Prevalence ####
prevAll <- read.csv("~/VA_IBD/llm_paper_tables_figures_sources_2025_04_22/resubmission_2025_06/csvs/prevalence_CDW_AND_MVP.csv")

#### Validation ####
val <- read.csv("~/VA_IBD/llm_paper_tables_figures_sources_2025_04_22/resubmission_2025_06/csvs/validation_CDW_AND_MVP.csv")

# Function for cohen's kappa on the entire dataset
cohen_kappa <- function(reviewer1, reviewer2) {
  # Check input
  if (length(reviewer1) != length(reviewer2)) {
    stop("Vectors must be of the same length")
  }

  # Convert to factors with same levels (ensures consistency)
  reviewer1 <- factor(reviewer1, levels = c(0, 1))
  reviewer2 <- factor(reviewer2, levels = c(0, 1))

  # Confusion matrix
  cm <- table(reviewer1, reviewer2)

  # Observed agreement
  po <- sum(diag(cm)) / sum(cm)

  # Expected agreement
  p1 <- rowSums(cm) / sum(cm)
  p2 <- colSums(cm) / sum(cm)
  pe <- sum(p1 * p2)

  # Kappa calculation
  kappa <- (po - pe) / (1 - pe)

  return(kappa)
}

cohen_kappa(val$BDJ, val$AD)

############# Get stats for Table S6 #############
llm_diff <- val[which((val$gemma2_9B_SPPO != val$gemma2_9B_SPPO_full | val$llama3_8B != val$llama3_8B_full)
                      & !is.na(val$llama3_8B_full) & !is.na(val$gemma2_9B_SPPO_full)),]
val_diff <- val[which((val$BDJ_full != val$consensusValidation)
                      & !is.na(val$llama3_8B_full) & !is.na(val$gemma2_9B_SPPO_full)),]

table(val$task[!is.na(val$llama3_8B_full) & !is.na(val$gemma2_9B_SPPO_full)])
table(llm_diff$task)
table(val_diff$task)
##################################################

# See how many times reviewers disagree
disagree <- val[which(val$BDJ != val$AD),]
nrow(disagree)/nrow(val[!is.na(val$AD),])

# Create vector for loop to process each cohort, task, source, model independently
val$task_cohort_source <- paste0(gsub("dys_", "dysplasia_", val$task, fixed = T), "_", val$cohort, "_", val$source)
uniq <- unique(val$task_cohort_source)
models <- unique(colnames(val)[1:6])
models <- c(models, "either", "both", "either_small", "both_small", "either_full", "both_full")
models <- c(models, "BDJ", "AD")

# Populate output data.frame
summary <- data.frame(
  task = NA, cohort = NA, source = NA,
  model = NA, conditionalModel = NA,
  model_prevalence_numerator = NA, model_prevalence_denominator = NA,
  model_prevalence_estimate = NA, f1 = NA, f1_lb_boot = NA, f1_ub_boot = NA,
  ppv = NA, ppv_lb_boot = NA, ppv_ub_boot = NA,
  npv = NA, npv_lb_boot = NA, npv_ub_boot = NA,
  recall = NA, recall_lb_boot = NA, recall_ub_boot = NA,
  specificity = NA, specificity_lb_boot = NA, specificity_ub_boot = NA,
  w = NA, w_lb_boot = NA, w_ub_boot = NA,
  mcc = NA, mcc_lb_boot = NA, mcc_ub_boot = NA,
  calibrated_f1 = NA, calibrated_f1_lb_boot = NA, calibrated_f1_ub_boot = NA,
  calibrated_precision = NA, calibrated_precision_lb_boot = NA, calibrated_precision_ub_boot = NA,
  N_val = NA, N_valPos = NA, N_valNeg = NA,
  N_modelPos_valPos_conditionalPos = NA, N_modelPos_valPos_conditionalNeg = NA,
  N_modelPos_valNeg_conditionalPos = NA, N_modelPos_valNeg_conditionalNeg = NA,
  N_modelNeg_valPos_conditionalPos = NA, N_modelNeg_valPos_conditionalNeg = NA,
  N_modelNeg_valNeg_conditionalPos = NA, N_modelNeg_valNeg_conditionalNeg = NA,
  BDJ_pos_AD_pos = NA, BDJ_pos_AD_neg = NA,
  BDJ_neg_AD_pos = NA, BDJ_neg_AD_neg = NA,
  cohensKappa = NA, cohensKappa_lb_boot=NA, cohensKappa_ub_boot=NA,
  median_nChar = NA, mean_nChar = NA
)

# Run analysis and bootstrap for each combination of task, cohort, source [i] and model [j].
for(i in 1:length(uniq)){
  all <- val[val$task_cohort_source==uniq[i],]
  for(j in 1:length(models)){
    
    # Get current model and prevalence
    model <- models[j]
    prevTmp <- prevAll[prevAll$task==all$task[i] & prevAll$cohort == all$cohort[i] & prevAll$source == all$source[i],]

    # Use the "_full" model answer for the full path report input
    if(grepl("full", model)){
      df <- all[!is.na(all$llama3_8B_full) & !is.na(all$gemma2_9B_SPPO_full),]
      df$consensusValidation <- df$BDJ_full # Use appropriate validation for full text
      nCharMedian <- median(df$ncharFull)
      nCharMean <- mean(df$ncharFull)
    }else{
      df <- all
      nCharMedian <- median(df$nchar)
      nCharMean <- mean(df$nchar)
    }

    # Set up for case of either/both
    if(model == "either" | model == "both"){
      model2 <- "llama3_8B"
      model1 <- "gemma2_9B_SPPO"
    }else if (model == "both_small" | model == "either_small"){
      model2 <- "llama3.2_3B"
      model1 <- "gemma2_2B"
    }else if (model == "both_full" | model == "either_full"){
      model2 <- "gemma2_9B_SPPO_full"
      model1 <- "llama3_8B_full"
    }else{
      model1 <- model
      model2 <- model
    }

    # Calculate prevalence of the models used for conditioning (Llama-3, except for IBD-CRC in MVP where either is used)
    w_llama <- prevTmp$N_pos_llama3_8B/prevTmp$N_llama3_8B
    w_either <- prevTmp$N_pos_either/prevTmp$N_overlap
    df$either <- df$llama3_8B==1 | df$gemma2_9B_SPPO == 1
    if(all(is.na(df$gemma2_9B_SPPO))){next}

    # Analyses must be run differently depending on whether the model was run for 
    # all reports or just the validation set. See Supplemetary section "Calculating performance metrics"
    if(uniq[i] == "invasive crc_ibd_MVP" | uniq[i] == "invasive crc unfiltered_ibd_MVP"){
      out <- compute_stats(df=df, prevTmp=prevTmp, model=model, model1=model1, model2=model2, w_conditional=w_either,
                           bootstrap=F, conditional = "either")
      conditionalModel <- "either"
      boot <- lapply(1:n_bootstrap, function(x){
        compute_stats(df=df, prevTmp=prevTmp, model=model, model1=model1, model2=model2, w_conditional=w_either,
                      bootstrap=T, conditional = "either")
      })
    }else{
      out <- compute_stats(df=df, prevTmp=prevTmp, model=model, model1=model1, model2=model2, w_conditional=w_llama,
                           bootstrap=F, conditional = "llama3_8B")
      conditionalModel <- "llama3_8B"
      boot <- lapply(1:n_bootstrap, function(x){
        compute_stats(df=df, prevTmp=prevTmp, model=model, model1=model1, model2=model2, w_conditional=w_llama, bootstrap=T, conditional = "llama3_8B")
      })
    }


    # Get bootstrap metrics (vectors of length 10k)
    f1_vec <- sapply(boot, function(x){x["f1"]})
    ppv_vec <- sapply(boot, function(x){x["ppv"]})
    npv_vec <- sapply(boot, function(x){x["npv"]})
    recall_vec <- sapply(boot, function(x){x["recall"]})
    specificity_vec <- sapply(boot, function(x){x["specificity"]})
    w_vec <- sapply(boot, function(x){x["w"]})
    mcc_vec <- sapply(boot, function(x){x["mcc"]})
    calf1_vec <- sapply(boot, function(x){x["calibrated_f1"]})
    calPPV_vec <- sapply(boot, function(x){x["calibrated_precision"]})
    cohensKappa_vec <- sapply(boot, function(x){x["cohensKappa"]})

    # Get 95% confidence intervals from the vectors
    ub <- 0.975
    lb <- 0.025
    
    # Save the data.frame
    summary <- rbind(summary, data.frame("task" = df$task[1], "cohort" = df$cohort[1], "source" = df$source[1],
                                         "model" = model, "conditionalModel" = conditionalModel,
                                         "model_prevalence_numerator" = out$prevNum,
                                         "model_prevalence_denominator" = out$prevDenom, "model_prevalence_estimate" = out$w,
                                         "f1" = out$f1, "f1_lb_boot" = quantile(f1_vec, lb, na.rm = TRUE), "f1_ub_boot" = quantile(f1_vec, ub, na.rm = TRUE),
                                         "ppv" = out$ppv, "ppv_lb_boot" = quantile(ppv_vec, lb, na.rm = TRUE), "ppv_ub_boot" = quantile(ppv_vec, ub, na.rm = TRUE),
                                         "npv" = out$npv, "npv_lb_boot" = quantile(npv_vec, lb, na.rm = TRUE), "npv_ub_boot" = quantile(npv_vec, ub, na.rm = TRUE),
                                         "recall" = out$recall, "recall_lb_boot" = quantile(recall_vec, lb, na.rm = TRUE), "recall_ub_boot" = quantile(recall_vec, ub, na.rm = TRUE),
                                         "specificity" = out$specificity, "specificity_lb_boot" = quantile(specificity_vec, lb, na.rm = TRUE), "specificity_ub_boot" = quantile(specificity_vec, ub, na.rm = TRUE),
                                         "w" = out$w, "w_lb_boot" = quantile(w_vec, lb, na.rm = TRUE), "w_ub_boot" = quantile(w_vec, ub, na.rm = TRUE),
                                         "mcc" = out$mcc, "mcc_lb_boot" = quantile(mcc_vec, lb, na.rm = TRUE), "mcc_ub_boot" = quantile(mcc_vec, ub, na.rm = TRUE),
                                         "calibrated_f1" = out$calibrated_f1, "calibrated_f1_lb_boot" = quantile(calf1_vec, lb, na.rm = TRUE), "calibrated_f1_ub_boot" = quantile(calf1_vec, ub, na.rm = TRUE),
                                         "calibrated_precision" = out$calibrated_precision, "calibrated_precision_lb_boot" = quantile(calPPV_vec, lb, na.rm = TRUE), "calibrated_precision_ub_boot" = quantile(calPPV_vec, ub, na.rm = TRUE),
                                         "N_val" = min(sum(!is.na(df[,model1])), sum(!is.na(df[, model2]))),
                                         "N_valPos" = sum(df$consensusValidation ==1),
                                         "N_valNeg" = sum(df$consensusValidation ==0),
                                         "N_modelPos_valPos_conditionalPos" = sum(out$model_bools & df$consensusValidation == 1 & df[,conditionalModel]),
                                         "N_modelPos_valPos_conditionalNeg" = sum(out$model_bools & df$consensusValidation == 1 & !df[,conditionalModel]),
                                         "N_modelPos_valNeg_conditionalPos" = sum(out$model_bools & df$consensusValidation == 0 & df[,conditionalModel]),
                                         "N_modelPos_valNeg_conditionalNeg" = sum(out$model_bools & df$consensusValidation == 0 & !df[,conditionalModel]),
                                         "N_modelNeg_valPos_conditionalPos" = sum(!out$model_bools & df$consensusValidation == 1 & df[,conditionalModel]),
                                         "N_modelNeg_valPos_conditionalNeg" = sum(!out$model_bools & df$consensusValidation == 1 & !df[,conditionalModel]),
                                         "N_modelNeg_valNeg_conditionalPos" = sum(!out$model_bools & df$consensusValidation == 0 & df[,conditionalModel]),
                                         "N_modelNeg_valNeg_conditionalNeg" = sum(!out$model_bools & df$consensusValidation == 0 & !df[,conditionalModel]),
                                         "BDJ_pos_AD_pos" = sum(df$BDJ==1 & df$AD==1),
                                         "BDJ_pos_AD_neg" = sum(df$BDJ==1 & df$AD==0),
                                         "BDJ_neg_AD_pos" =  sum(df$BDJ==0 & df$AD==1),
                                         "BDJ_neg_AD_neg" = sum(df$BDJ==0 & df$AD==0),
                                         "cohensKappa" = out$cohensKappa, "cohensKappa_lb_boot" = quantile(cohensKappa_vec, lb, na.rm = TRUE), "cohensKappa_ub_boot" = quantile(cohensKappa_vec, ub, na.rm = TRUE),
                                         "median_nChar" = nCharMedian,
                                         "mean_nChar" = nCharMean))
  }
}

# Remove NA and rename to df
summary <- summary[!is.na(summary$model_prevalence_estimate) &
                     !is.na(summary$task),]
df <- summary

# Get stats
df$N_modelPos <- rowSums(df[,grepl("N_modelPos", colnames(df))])
df$N_modelNeg <- rowSums(df[,grepl("N_modelNeg", colnames(df))])

# Calculate analytical confidence intervals
df$ppv_lowerBound_analytical <- unlist(mapply(function(x, y){binom.test(x, y)$conf.int[1]}, x=round(df$ppv*df$N_modelPos), y=df$N_modelPos))
df$ppv_upperBound_analytical <- unlist(mapply(function(x, y){binom.test(x, y)$conf.int[2]}, x=round(df$ppv*df$N_modelPos), y=df$N_modelPos))
df$npv_lowerBound_analytical <- unlist(mapply(function(x, y){binom.test(x, y)$conf.int[1]}, x=round(df$npv*df$N_modelNeg), y=df$N_modelNeg))
df$npv_upperBound_analytical <- unlist(mapply(function(x, y){binom.test(x, y)$conf.int[2]}, x=round(df$npv*df$N_modelNeg), y=df$N_modelNeg))

# Take the min LB and max UB and use that
df$ppv_lowerBound <- pmin(df$ppv_lowerBound_analytical, df$ppv_lb_boot)
df$npv_lowerBound <- pmin(df$npv_lowerBound_analytical, df$npv_lb_boot)
df$ppv_upperBound <- pmax(df$ppv_upperBound_analytical, df$ppv_ub_boot)
df$npv_upperBound <- pmax(df$npv_upperBound_analytical, df$npv_ub_boot)

# Set the row ordering in excel based on this variable (CRC then HGD/CRC then Dysplasia then IND)
df$hidden_sort <- 1
df$hidden_sort[grepl("hgd", df$task)] <- 2
df$hidden_sort[grepl("dys", df$task)] <- 3
df$hidden_sort[grepl("indefinite", df$task)] <- 4

# Make polished excel column names
df$Task <- "CRC"
df$Task[grepl("hgd", df$task)] <- "HGD/CRC"
df$Task[grepl("dys", df$task)] <- "Dysplasia"
df$Task[grepl("indefinite", df$task)] <- "IND"
df$Task[grepl("unfiltered", df$task)] <- "CRC (unfiltered)"

df$Cohort <- "IBD"
df$Cohort[grepl("non", df$cohort)] <- "non-IBD"

df$Source <- toupper(df$source)
df$Model <- df$model
df$Input <- "Microscopic exam"
df$Input[grepl("full", df$model)] <- "Full pathology report"

df$`Model prevalence estimate` <- sprintf("%.3f", df$w)
df$`PPV (LB - UB)` <- paste0(sprintf("%.3f", df$ppv), " (", sprintf("%.2f", df$ppv_lowerBound), " - ", sprintf("%.2f", df$ppv_upperBound), ")")
df$`NPV (LB - UB)` <- paste0(sprintf("%.3f", df$npv), " (", sprintf("%.2f", df$npv_lowerBound), " - ", sprintf("%.2f", df$npv_upperBound), ")")
df$PPVc <- sprintf("%.3f", df$calibrated_precision)
df$`Recall (Sensitivity)` <- sprintf("%.3f", df$recall)
df$Specificity <- sprintf("%.3f", df$specificity)
df$F1 <- paste0(sprintf("%.3f", df$f1))
df$F1c <- paste0(sprintf("%.3f", df$calibrated_f1))
df$MCC <- sprintf("%.3f", df$mcc)
df$`Cohen's kappa` <- sprintf("%.3f", df$cohensKappa)

# Save as csv, then make tables in excel from this source (save as .xlsx)
write.csv(df, paste0("~/VA_IBD/llm_paper_tables_figures_sources_2025_04_22/resubmission_2025_06/csvs/all_results_", Sys.Date(), ".csv"), row.names=F)

