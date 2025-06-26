# Function to compute stats and bootstrap given the model to be evaluated,
# the model used for conditioning (i.e., stratified sampling) and the
# validation.
compute_stats <- function(df, prevTmp, model, model1, model2, w_conditional, bootstrap=F, conditional="llama3_8B"){

  # For bootstrapping, sample w/ replacement
  if(bootstrap){
    df <- df[sample(1:nrow(df), nrow(df), replace=T),]
  }

  # Because "either" is used as conditional, make sure we have it in df
  df$either <- df$llama3_8B | df$gemma2_9B_SPPO

  # Make sure that number run for Gemma-2 matches the number run for both models (N_overlap)
  stopifnot(prevTmp$N_gemma2_9B_SPPO==prevTmp$N_overlap)

  # These are the model runs where we've run the model on "N_overlap" unconditioned
  # reports. Other models were only run for the validation set and must be treated differently.
  if(model %in% c("gemma2_9B_SPPO", "either", "both", "llama3_8B") & conditional %in% c("either", "llama3_8B")){

    # Get overall prevalence of the model
    prevNum <- prevTmp[,paste0("N_pos_", model)]
    if(model %in% c("either", "both")){
      prevDenom <- prevTmp$N_overlap
    }else{
      prevDenom <- prevTmp[,paste0("N_", model)]
    }

    # Get model prevalence (w) and model answers (model_bools)
    w <- prevNum/prevDenom
    df$both <- df$llama3_8B & df$gemma2_9B_SPPO
    model_bools <- df[,model]

    # From here on, we care only about the set where we have info on both models (N_overlap)
    if(prevTmp$N_overlap == prevTmp$N_llama3_8B){
      # Sanity check to make sure we're calculating N_pos for llama-3 correctly
      stopifnot(prevTmp$N_pos_llama3_8B == prevTmp$N_pos_either - prevTmp$N_pos_gemma2_9B_SPPO + prevTmp$N_pos_both)
    }else{
      prevTmp$N_pos_llama3_8B <- prevTmp$N_pos_either - prevTmp$N_pos_gemma2_9B_SPPO + prevTmp$N_pos_both
      prevTmp$N_llama3_8B <- prevTmp$N_overlap
    }

    # Of the model positives, what fraction occur when the conditional is positive?
    # This will be used to weight the PPV accordingly
    if(conditional == "either"){
      weight_for_ppv <- 1 # All model positives are also conditional positive when conditional is "either"
    }else{
      # If conditional is not "either", conditional must be Llama-3
      # If model is also Llama-3, weight_for_ppv becomes irrelevant because ppv2 is NA
      weight_for_ppv <- prevTmp$N_pos_both/prevTmp[,paste0("N_pos_", model)]
    }

    # Calculate the PPV when conditional model is TRUE. Weight accordingly.
    # WIP: NEED TO FIX TO FIX LLAMA-3 STATS WHEN CONDITIONAL IS EITHER
    ppv1 <- sum(df$consensusValidation==1 & model_bools & df[,conditional])/
      sum(model_bools & df[,conditional])*
      weight_for_ppv # Weight by how often this occurs

    # Calculate the PPV when conditional model is FALSE
    # This will be division by zero if conditional is "either". Handled in if statements below.
    ppv2 <- sum(df$consensusValidation==1 & model_bools & !df[,conditional])/
      sum(model_bools & !df[,conditional])*
      (1 - weight_for_ppv) # Weight by how often this occurs


    if(is.na(ppv1)){ # Should never happen
      stop("PPV1 should not be NA")

    } else if (is.na(ppv2)){ # Okay if this happens
      ppv <- sum(df$consensusValidation==1 & model_bools) / sum(model_bools)

    } else{
      if(model == conditional){
        stop("When model and conditional are the same, ppv2 should be NA")
      }else if(conditional == "either"){
        stop("When conditional is either, ppv2 should be NA")
      }
      ppv <- sum(c(ppv1, ppv2))
    }

    # Of the model negatives, what fraction occur when the conditional is positive?
    # This will be used to weight the NPV accordingly
    if(conditional == "llama3_8B" & model == "both"){
      weight_for_npv <- (prevTmp$N_pos_llama3_8B - prevTmp$N_pos_both)/
        (prevTmp$N_overlap - prevTmp[,paste0("N_pos_", model)])
    }else{
      weight_for_npv <- (prevTmp$N_pos_either - prevTmp[,paste0("N_pos_", model)])/
        (prevTmp$N_overlap - prevTmp[,paste0("N_pos_", model)])
    }

    npv1 <- sum(!model_bools & df$consensusValidation==0 & df[,conditional])/
      sum(!model_bools & df[,conditional])*
      weight_for_npv

    npv2 <- sum(!model_bools & df$consensusValidation==0 & !df[,conditional])/
      sum(!model_bools & !df[,conditional])*
      (1-weight_for_npv)

    if(is.na(npv1)){ # Will happen
      npv <- sum(!model_bools & df$consensusValidation==0) / sum(!model_bools)

    } else if (is.na(npv2)){ # Should never happen!
      stop("NPV2 should not be NA")

    } else{
      if(model == conditional){
        stop("When model and conditional are the same, npv1 should be NA")
      }
      npv <- sum(c(npv1, npv2))
    }

  }else{

    if(grepl("either", model)){
      model_bools <- df[,model1] | df[,model2]
    }else{
      model_bools <- df[,model1] & df[,model2] # if only one model, model1 = model2 and this still works
    }

    # Can't calculate prevalence directly b/c we didn't run on all
    prevNum <- NA
    prevDenom <- NA
    w <- sum(model_bools & df[,conditional])/sum(df[,conditional])*w_conditional +
      sum(model_bools & !df[,conditional])/sum(!df[,conditional])*(1-w_conditional)

    # Calculate the expected fraction of TP's when Llama-3 is TRUE
    TP1 <- sum(df$consensusValidation==1 & df[,conditional] & model_bools)/
      sum(df[,conditional])*w_conditional

    # Calculate the expected fraction of TP's when Llama-3 is FALSE
    TP2 <- sum(df$consensusValidation==1 & !df[,conditional] & model_bools)/
      sum(!df[,conditional])*(1-w_conditional)

    # Divide the expected fraction of TP by the total expected fraction of model positives (w = TP + FP)
    ppv <- sum(c(TP1, TP2))/w

    if(is.na(ppv) & !any(is.na(model_bools))){stop("PPV is NA")}

    TN1 <- sum(df$consensusValidation==0 & df[,conditional] & !model_bools)/
      sum(df[,conditional])*w_conditional

    TN2 <- sum(df$consensusValidation==0 & !df[,conditional] & !model_bools)/
      sum(!df[,conditional])*(1-w_conditional)

    npv <- sum(c(TN1, TN2))/(1-w)
    if(is.na(npv) & !any(is.na(model_bools))){stop("NPV is NA")}

  }

  # Recall
  recall <- w*ppv / (w*ppv + (1-w)*(1-npv))

  # Specificity
  specificity <- (1-w)*(npv)/((1-w)*(npv)+w*(1-ppv))

  # F1 Score (Harmonic mean of Precision and Recall)
  F1 <- 2 * ((ppv * recall) / (ppv + recall))

  # Matthew's correlation coefficient
  MCC <- sqrt(ppv*recall*specificity*npv) - sqrt((1-ppv)*(1-recall)*(1-specificity)*(1-npv))

  # Calibrated stats
  pi0 <- 0.5
  calibrated_precision <- 1 / ( 1 + (w*(1-pi0)*(1-ppv)*w) /
                                     (pi0*(1-w)*(w*ppv + (1-w)*(1-npv))))

  calibrated_f1 <- 2 * ((calibrated_precision * recall) / (calibrated_precision + recall))

  # Cohen's kappa
  BDJ_pos_AD_pos = sum(df$BDJ==1 & df$AD==1)
  BDJ_pos_AD_neg = sum(df$BDJ==1 & df$AD==0)
  BDJ_neg_AD_pos =  sum(df$BDJ==0 & df$AD==1)
  BDJ_neg_AD_neg = sum(df$BDJ==0 & df$AD==0)
  cohensKappa <- 2*(BDJ_pos_AD_pos*BDJ_neg_AD_neg - BDJ_neg_AD_pos*BDJ_pos_AD_neg) /
    ((BDJ_pos_AD_pos+BDJ_pos_AD_neg)*(BDJ_pos_AD_neg+BDJ_neg_AD_neg) +
       (BDJ_pos_AD_pos+BDJ_neg_AD_pos)*(BDJ_neg_AD_pos+BDJ_neg_AD_neg))

  if(bootstrap){
    (return(c("w"=w, "ppv"=ppv, "npv"=npv, "recall"=recall, "specificity"=specificity, "f1"=F1, "mcc"=MCC,
              "calibrated_precision" = calibrated_precision, "calibrated_f1" = calibrated_f1, "cohensKappa" = cohensKappa)))
  }else{
    return(list("prevNum" = prevNum, "prevDenom"=prevDenom,
          "w"=w, "ppv"=ppv, "npv"=npv, "recall"=recall, "specificity"=specificity, "f1"=F1, "mcc"=MCC,
                "calibrated_precision" = calibrated_precision, "calibrated_f1" = calibrated_f1,
          "cohensKappa" = cohensKappa,
                "model_bools" = model_bools))
  }
}
