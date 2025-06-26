# Reproducing Paper Results

Using the two R scripts and two of the three CSV files (`prevalence_CDW_AND_MVP.csv` and `validation_CDW_AND_MVP.csv`), you can reproduce the performance metric calculations (the third CSV included) using our stratified sampling approach with bootstrapping.

## Instructions

1. **Download the Code:**  
   Download the `reproduce_paper_results` directory.

2. **Set File Paths:**  
   Modify the paths in the R script `make_csv_from_raw_data.R` to match your workspace.

3. **Run the Script:**  
   Run the R script. This will produce an `all_results_{DATE}.csv` file that should match the included `all_results.csv`, except for minor random variation introduced by bootstrapping.

## Output Files

- **all_results.csv:**  
  Contains the data used to generate most of the tables in the paper.  
  (For the paper, we saved this as an Excel file, hid some rows/columns, and adjusted formatting to create the final tables.)

- **all_results.xlsx:**  
  Contains the tables in Excel format for easy interaction with the data.