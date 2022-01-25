# serpens_icemap - Routines to model the stellar population and extinction in Serpens Main field.

## Intro
This python project began as my feasibility study of NIRSpec MSA observations to map ice absorption in YSO envelopes, using background stars in the Serpens Main field. After the proposal was successful (PID 1611), the code has been converted to something readable and useable by someone other than just myself. 

Implementation of MIRAGE for simulating our NIRCam pre-imaging data is underway and should be pushed soon.

## Getting Started
After cloning this repo, it would be easiest to creat a new conda environment using the included environment.yml file:

```
conda env create -f environment.yml
```

This should create an environment named `serpens` which should support the running of this code.

Next, you'll need to run `python dustmaps_config.py` to download the Schlegel, Finkbeiner and Davis dust map for use as a foreground extinction estimator.
