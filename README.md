# summer23_aylab

This repository contains information about the codebase and how to use it.

## Prerequisites
### Dependencies – Turing Cluster  

If you will be running this code on Colgate's Turing Cluster, it (at the time of writing this) already has a shared `AyLab` directory, which contains a conda installation with all the required dependencies for this codebase. Make sure it is activated before you run any code.
```
$ source /shared/devenvs/AyLab/conda/bin/activate
```
You can check which conda installation you are using at any time with the command `which conda`.  
### Dependencies – Personal Device

If you will be running this code on your personal device, the dependencies (at least the ones I remember lol) are listed in `requirements.txt`. Make sure to install them before you run any code. Some of them might need to be installed through different channels.  
The `ukbb_parser` module (not listed in `requirements.txt`) is not to be installed through any package manager. Instead, follow the steps at https://github.com/nadavbra/ukbb_parser/tree/master#installation.
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

### Accessing UKBB Genetic Data – Cluster

***TODO: IMPLEMENT THIS PART***

### Accessing Genetic UKBB Data – Personal Device (NOT RECOMMENDED)

***TODO: IMPLEMENT THIS PART***

## additional resources in this repository  
Check `reusable_code.md` for functions from Hieu and Cole's code that we might use.