# summer23_aylab

This repository contains information about the codebase and how to use it.

## Prerequisites

### Installing UKBB Helper Programs

UK Biobank has several helper programs that are required to work with UKBB data (more information on https://biobank.ctsu.ox.ac.uk/~bbdatan/Data_Access_Guide_v3.1.pdf. There might be a newer version when you are reading this, check at https://biobank.ctsu.ox.ac.uk/crystal/exinfo.cgi?src=accessing_data_guide)

1. Make a directory for the UKBB utilities at the root of the repository. Directory name `ukbb_util` is already included in `.gitignore`, but feel free to make your own adjustments.

```
mkdir ukbb_util
cd ukbb_util
```
2. Install `ukmd5` to calculate size and checksum of a file.
```
wget  -nd  biobank.ndph.ox.ac.uk/ukb/util/ukbmd5
chmod 755 ukbmd5
```
3. Install `ukbunpack` to unpack downloaded data files.
```
wget  -nd  biobank.ndph.ox.ac.uk/ukb/util/ukbunpack
chmod 755 ukbunpack
```
4. Install `ukbconv` to convert unpacked filed into other formats.
```
wget  -nd  biobank.ndph.ox.ac.uk/ukb/util/ukbconv
chmod 755 ukbconv
```
### Moving and Unpacking your Data
1. Make a directory for data at the root of the repository. Again, directory name `data` is already included in `.gitignore`, but feel free to make your own adjustments.
```
mkdir data
cd data
```
2. Using your preferred method (VSCode, Cyberduck, etc.), move your encrypted data to the directory you just created.
3. The following steps are not required if your data is already in the desired format (no `.enc` extension)

## additional resources in this repository  
Check `reusable_code.md` for functions from Hieu and Cole's code that we might use.