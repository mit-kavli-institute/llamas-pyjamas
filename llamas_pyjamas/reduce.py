
import os
import argparse
import pickle


from llamas_pyjamas.Trace.traceLlamasMaster import run_ray_tracing
from llamas_pyjamas.config import BASE_DIR, OUTPUT_DIR, DATA_DIR, CALIB_DIR, BIAS_DIR, LUT_DIR
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas
import llamas_pyjamas.GUI.guiExtract as ge
from llamas_pyjamas.File.llamasIO import process_fits_by_color
import llamas_pyjamas.Arc.arcLlamas as arc



_linefile = os.path.join(LUT_DIR, '')

### This needs to be edited to handle a trace file per channel
def generate_traces(red_flat, green_flat, blue_flat, output_dir, bias=None):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    assert os.path.exists(red_flat), "Red flat file does not exist."
    assert os.path.exists(green_flat), "Green flat file does not exist."
    assert os.path.exists(blue_flat), "Blue flat file does not exist."

    run_ray_tracing(red_flat, outpath=output_dir, channel='red', use_bias=bias)
    run_ray_tracing(green_flat, outpath=output_dir, channel='green', use_bias=bias)
    run_ray_tracing(blue_flat, outpath=output_dir, channel='blue', use_bias=bias)
    print(f"Traces generated and saved to {output_dir}")

    return


###need to edit GUI extract to give custom output_dir
def extract_flat_field(flat_file_dir, output_dir, bias_file=None):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ge.GUI_extract(flat_file_dir, output_dir=output_dir, use_bias=bias_file)

    return


def run_extraction(science_file, bias_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    assert os.path.exists(science_file), "Science file does not exist."

    ge.GUI_extract(science_file, output_dir=output_dir, use_bias=bias_file)

    return


def calc_wavelength_soln(arc_file, output_dir, bias=None):

    ge.GUI_extract(arc_file, use_bias=bias, output_dir=output_dir)

    arc_picklename = os.path.join(output_dir, os.path.basename(arc_file).replace('_mef.fits', '_extract.pkl'))

    with open(arc_picklename, 'rb') as fp:
        batch_data = pickle.load(fp)
    
    arcdict = ExtractLlamas.loadExtraction(arc_picklename)
    arcspec, metadata = arcdict['extractions'], arcdict['metadata']

    arc.shiftArcX(arc_picklename)

    return arcdict



def relative_throughput(shift_picklename, flat_picklename):

    arc.fiberRelativeThroughput(flat_picklename, shift_picklename)
    ### need to add code in to return the name of the throughput file
    return


def correct_wavelengths(arcdict):

    arcdict = ExtractLlamas.loadExtraction(os.path.join(LUT_DIR, 'LLAMAS_reference_arc.pkl'))
    
    return



def construct_cube():
    ###need to make a new class for this bit
    return

def main(config_file):
    print("This is a placeholder for the reduce module.")
    # You can add functionality here as needed.
    with open(config_file, 'rb') as f:
        config = pickle.load(f)
    print(f"Loaded configuration from {config_file}")
    
    try:
        generate_traces()

        extract_flat_fields()
        run_extraction()
        relative_throughput()
        correct_wavelengths()
        construct_cube()
    except Exception as e:
        print(f"An error occurred: {e}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduce module placeholder")
    parser.add_argument('config_file', type=str, help='Path to the configuration file')
    args = parser.parse_args()
    
    
    main(args.config_file)