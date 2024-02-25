
import sys
import numpy as np
from nnenum.enumerate import enumerate_network
from nnenum.settings import Settings
from nnenum.result import Result
from nnenum.src.nnenum.nnenum import make_spec, set_control_settings, set_exact_settings, set_image_settings
from nnenum.src.nnenum.onnx_network import load_onnx_network, load_onnx_network_optimized

def verify_onnx_with_lib(onnx_filename, vnnlib_filename, 
                         timeout=60, processes=1, 
                         settings_str="auto", outfile=None):
    Settings.NUM_PROCESSES = processes
    settings_str = "auto"
    spec_list, input_dtype = make_spec(vnnlib_filename, onnx_filename)
    
    try:
        network = load_onnx_network_optimized(onnx_filename)
    except:
        # cannot do optimized load due to unsupported layers
        network = load_onnx_network(onnx_filename)
    result_str = 'none' # gets overridden

    num_inputs = len(spec_list[0][0])
    if settings_str == "auto":
        if num_inputs < 700:
            set_control_settings()
        else:
            set_image_settings()
    elif settings_str == "control":
        set_control_settings()
    elif settings_str == "image":
        set_image_settings()
    else:
        assert settings_str == "exact"
        set_exact_settings()

    for init_box, spec in spec_list:
        init_box = np.array(init_box, dtype=input_dtype)

        if timeout is not None:
            if timeout <= 0:
                result_str = 'timeout'
                break

            Settings.TIMEOUT = timeout

        res = enumerate_network(init_box, network, spec)
        result_str = res.result_str

        if timeout is not None:
            # reduce timeout by the runtime
            timeout -= res.total_secs

        if result_str != "safe":
            break

    # rename for VNNCOMP21:
        
    if result_str == "safe":
        result_str = "holds"
    elif "unsafe" in result_str:
        result_str = "violated"

    if outfile is not None:
        with open(outfile, 'w') as f:
            f.write(result_str)
    #print(result_str)

    if result_str == 'error':
        sys.exit(Result.results.index('error'))
        
if __name__ == "__main__":
    onnx_file = "Fed-Time-Series-Forecasting/super_resolution.onnx"
    
    
