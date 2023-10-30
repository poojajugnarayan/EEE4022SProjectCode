# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/python/reconstruction_system/run_system.py

import json
import time
import datetime
import os, sys
import open3d as o3d

pyexample_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pyexample_path)


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from initialize_config import initialize_config, dataset_loader

if __name__ == "__main__":


    # load dataset and check folder structure
    # config = dataset_loader("lounge")

    with open('3Dconfig.json','r') as json_file:
        config = json.load(json_file)
        initialize_config(config)

    assert config is not None

    config['debug_mode'] = False

    config['device'] = 'cpu:0'

    print("====================================")
    print("Configuration")
    print("====================================")
    for key, val in config.items():
        print("%40s : %s" % (key, str(val)))

    times = [0, 0, 0, 0, 0, 0]
    start_time = time.time()
    import make_fragments
    make_fragments.run(config)
    times[0] = time.time() - start_time
    start_time = time.time()
    import register_fragments
    register_fragments.run(config)
    times[1] = time.time() - start_time
    start_time = time.time()
    import refine_registration
    refine_registration.run(config)
    times[2] = time.time() - start_time
    start_time = time.time()
    import integrate_scene
    integrate_scene.run(config)
    times[3] = time.time() - start_time
 
 

    print("====================================")
    print("Elapsed time (in h:m:s)")
    print("====================================")
    print("- Making fragments    %s" % datetime.timedelta(seconds=times[0]))
    print("- Register fragments  %s" % datetime.timedelta(seconds=times[1]))
    print("- Refine registration %s" % datetime.timedelta(seconds=times[2]))
    print("- Integrate frames    %s" % datetime.timedelta(seconds=times[3]))
    print("- Total               %s" % datetime.timedelta(seconds=sum(times)))
    sys.stdout.flush()
# "segmented_1/segmented_", 
# "images_1/image_", 