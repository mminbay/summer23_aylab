# summer23_aylab

This repository contains information about the codebase utilized in the <a href="https://doi.org/10.1016/j.psychres.2024.115948" target="_blank">UK Biobank Depression Study</a>, and how to use it.
## Prerequisites
### Dependencies – Turing Cluster  

If you will be running this code on Colgate's Turing Cluster, it (at the time of writing this) already has a shared `AyLab` directory, which contains a conda installation with all the required dependencies for this codebase. Make sure it is activated before you run any code.
```
$ source /datalake/AyLab/conda_dev_envs/summer23_env/bin/activate
```
You can check which conda installation you are using at any time with the command `which conda`.
### Installing UKBB Helper Programs

UK Biobank has several helper programs that are required to work with UKBB data (more information on https://biobank.ctsu.ox.ac.uk/~bbdatan/Data_Access_Guide_v3.1.pdf. There might be a newer version when you are reading this, check at https://biobank.ctsu.ox.ac.uk/crystal/exinfo.cgi?src=accessing_data_guide)

1. Make a directory for the UKBB utilities at the root of the repository. Directory name `ukbb_util` is already included in `.gitignore`, but feel free to make your own adjustments.

```
$ mkdir ukbb_util
$ cd ukbb_util
```
2. Install `ukmd5` to calculate size and checksum of a file.
```
$ wget  -nd  biobank.ndph.ox.ac.uk/ukb/util/ukbmd5
$ chmod 755 ukbmd5
```
3. Install `ukbunpack` to unpack downloaded data files.
```
$ wget  -nd  biobank.ndph.ox.ac.uk/ukb/util/ukbunpack
$ chmod 755 ukbunpack
```
4. Install `ukbconv` to convert unpacked filed into other formats.
```
$ wget  -nd  biobank.ndph.ox.ac.uk/ukb/util/ukbconv
$ chmod 755 ukbconv
```
### Moving and Unpacking your UKBB Data

1. Make a directory for data at the root of the repository. Again, directory name `data` is already included in `.gitignore`, but feel free to make your own adjustments.
```
$ mkdir data
$ cd data
```
2. Using your preferred method (VSCode, Cyberduck, etc.), move your encrypted data to the directory you just created.  

The following steps are not required if your data is already in the desired format (no `.enc` extension)  

3. Using your preferred method (VSCode, Cyberduck, etc.), move the `.key` file that you received with the download to the directory you just created.  
4. Use `ukbunpack` to unpack the data file you have uploaded. Change the filenames below accordingly for your case. This step should create a `.enc_ukb` file.
```
$ cd ../ukbb_util
$ ./ukbunpack ../data/<data_file>.enc ../data/<key_file>.key
```
5. Use `ukbconv` to convert the unpacked data to `.csv` format (or any other format you will work with: check link provided above for additional options).
```
$ ./ukbconv ../data/<data_file>.enc_ukb csv
```  
Feel free to delete the resultant `.log` file and reorganize your `data` folder as you wish after this step.

### Relocating `.ukbb_paths.py`

The `ukbb_parser` module assumes the existence of a `.ukbb_paths.py` file at **the root of your user directory** (`~/.ukbb_paths.py`). Move this file to the appropriate directory and follow comments on the file to make the necessary changes.

### Downloading and Accessing UKB Genetic Data

The UK Biobank data is available for a fee. For reference to their data and data fields, you can look at https://biobank.ctsu.ox.ac.uk/crystal/label.cgi?id=263.  

You will need `gfetch` to download files as described in the above resource. 
```
$ cd /whichever/directory/you/will/call/gfetch/from/
$ wget  -nd  biobank.ndph.ox.ac.uk/ukb/util/gfetch
```
Make sure to remove the participants who have withdrawn.

### Configuring `.env`  
Since you might want to organize your `data` folder as you wish, there is a `.env.example` provided in this repository. Set the environment variables as appropriate and rename the file to `.env` to make it usable.

## Usage
### `load_data.py`
This file defines the `DataLoader` class, which is meant to help you compile a final dataset from your phenotype data and genetic data.

### `feature_selector.py`
This file defines the `FeatureSelector` class, alongside with some feature selection functions that will be used by this class. Make sure you are importing all of these functions (for now these are `chisquare`, `infogain`, `jmi`, `mrmr`, and `mann_whitney_u`). 

(Hopefully) the only method you should have to interact with here is `bootstrapped_feat_select()`.

If you are also working with a large dataset, you might have your SNP data spread across multiple files and call a separate `FeatureSelector` for all of them, meaning you will have different output files at the end of your feature selection. If you want to compile your selected SNPs into a single .csv file for further analysis (or for subset based feature selection), this file also contains a `compile_snps()` function that does exactly that.
