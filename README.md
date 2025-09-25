# LlamasPipeline

This repository contains Data Reduction Pipeline tools for the LLAMAS Integral Field Spectrometer

It is being updated frequently as the instrument comes online.

THIS IS AN IN-PROGRESS DEVELOPMENT PIPELINE.  IT IS MADE AVAILABLE TO THE PUBLIC IN READ-ONLY FORMAT
FOR OBSERVATION PLANNING AND EXECUTION, BUT NO WARRANTY IS MADE REGARDING INSTALLATION OR 
EXECUTION UNTIL INSTRUMENT COMMISSIONING IS COMPLETE.

**Users of this pipeline are requested to cite Hughes et al. (in prep)**.

<details>
<summary>Citation</summary>
```bibtex
@unpublished{Hughes2025,
  author       = {Hughes, Sarah and others},
  title        = {{The LLAMAS data reduction pipeline}},
  note         = {in preparation},
  year         = {2025}}
```
</details>

For instructions on installation, compilation, and runtime, please see below and the files in the Tutorials directory. Instructions will be kept as up to date as possible as the pipeline develops.

**If you are reducing data from the Nov/Dec 2024 commissioning run, please contact me directly at slhughes@mit.edu for additional support to reduce your observations**

Information regarding updates will be sent via email to those interested in using the mailing list below.

To join the mailing list:
https://mailman.mit.edu/mailman/listinfo/llamas-pipeline

or send a blank email to llamas-pipeline-join@mit.edu

You will then receive an email asking you to confirm your subscription request. Once completed, you will be subscribed to our mailing list.

## Installation instructions

To install the pipeline, first clone the repository and `cd` into the llamas-pyjamas directory that contains the README.md and poetry files

**This pipeline runs on python 3.12. we recommend making a new virtual environment prior to installation:** `conda create -n myenv python=3.12`

Run the command `pip install -e .` to begin the installation process. **Some additional packages may need to be pip-installed as development continues**

### Auxiliary Files

To run the current pipeline, two steps are required. First, **please download the mastercalib files from this location and have them in a folder named 'mastercalib' located within the llamas_pyjamas subfolder**: https://mit-kavli.box.com/s/bath5hhtjqsn3m89l7579u1ev4rk2xjo

You will also need to download the combined_bias.fits file and keep a copy within both llamas_pyjamas and the Bias subfolder to run the Quicklook GUI.

Secondly, **download the wavelength solution file and place it in the LUT subfolder**, which can be downloaded from here: https://mit-kavli.box.com/s/v4kwlsx02nevnv5nw3p1i58lsxxoowe6

The final structure of your repo should look like this to run both the reduction script and the Quick Look GUI:

```
llamas-pyjamas/
└── llamas_pyjamas/
    │   └── combined_bias.fits
    ├── Arc/
    ├── Bias/
    │   └── combined_bias.fits
    ├── Cube/
    ├── Docs/
    ├── Extract/
    ├── File/
    ├── Flat/
    ├── Flux/
    ├── GUI/
    ├── Image/
    ├── LUT/
    │   └── LLAMAS_reference_arc.pkl
    ├── mastercalib/
    │   └── combined_bias.fits
    │   └── LLAMAS*trace.pkl files
    ├── Postprocessing/
    ├── QA/
    ├── example_config.txt
    ├── Trace/
    ├── Tutorials/
    └── Utils/
```
### Reduction script

To perform an end-to-end reduction of your data, use the example_config.txt file as guidance to construct your own configuration file. 

This config file uses paths to specify which calibration images should be used and which science files should be reduced. **List the file paths but do not put them in quotations**

**Science files to be reduced can be done as a batch process but will all use the same calibration files listed**

Sky subtraction and flux calibration are not currently implemented in the reduction process but will be added in future. 

To run the script, first `cd llamas-pyjamas/llamas_pyjamas` and activate your Python environment where the pipeline is installed. Then run the command `python reduce.py 'path/to/your/config.txt'` to initiate the data reduction process.

The speed of reduction will vary depending on your machine specifications. If errors occur, there are log files produced within the Utils folder that can be helpful for diagnosing issues.


### QuickLook Demo files
To test run the quick look pipeline on data, we have provided a standard star raw image, flat field images, and a bias image to create an extracted WhiteLight image. https://mit-kavli.box.com/s/k7s3bmwu98q3iljm4djpzidxj7qg26cl
