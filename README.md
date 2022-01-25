# serpens_icemap - Routines to model the stellar population and extinction in Serpens Main field.

## Intro
This python project began as my feasibility study of NIRSpec MSA observations to map ice absorption in YSO envelopes, using background stars in the Serpens Main field. After the proposal was successful (PID 1611), the code has been converted to something readable and useable by someone other than just myself. 

Implementation of MIRAGE for simulating our NIRCam pre-imaging data is underway and should be pushed soon.

## Getting Started
After cloning this repo, it would be easiest to creat a new conda environment using the included environment.yml file:

```
conda env create --file environment.yml
```

This will create an environment named `serpens` which should support the running of this code. Enter this environment with `conda activate serpens`.

Next, you'll need to run `python dustmaps_config.py` to download the Schlegel, Finkbeiner and Davis dust map for use as a foreground extinction estimator.

If you have not installed pysynphot/synphot/stsynphot locally before, there will be some configuration required for synphot and stsynphot. You can follow the documentation provided by those packages, but the key details to follow are setting the environment variable `PYSYN_CDBS`, then retrieving the Castelli-Kurucz stellar models, as well as default calibration files.

To derive the weights for target selection in APT, the code uses a scaled map of the 160 micron flux from Herschel PACS. This file can be retrieved from the Herschel Science Archive (http://archives.esac.esa.int/hsa/whsa/) by searching for the Observation ID 1342226695.



