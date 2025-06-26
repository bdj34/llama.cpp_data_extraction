Using the two R scripts and two of the three CSVs ("prevalence_CDW_AND_MVP.csv" and "validation_CDW_AND_MVP.csv"), you can rreproduce the performance metric calculation (the third csv included) using our stratified sampling approach with bootstrapping.

To do this, download the "reproduce_paper_results" directory and modify the paths in the R script "make_csv_from_raw_data.R" to match your workspace. Then, simply run the R script and you will produce an all_results_{DATE}.csv" file that should match exactly with the "all_results.csv" file included, up to random variation introduced by the bootstrapping.

The "all_results.csv" contains the data we used to make most of the tables in the paper. We simply saved it as an excel file, hid some rows/cols and modified the formatting to produce the tables.

The "all_results.xlsx" file contains the tables in excel format, for easily interacting with the data.
