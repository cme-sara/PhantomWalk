import pytest
import datetime
import os
from contextlib import suppress

import phantomwalk.lib.create_system_dpd as dpd

# helper to not have several repetitive try/catch blocks for cleanup
def rm_files(*files):
    for fl in files:
        try:
            os.remove(fl)
        except:
            pass

def test_creation():
    loop_timeout = 60
    snap, closest, time, e_cut = dpd.create_polymer_system_dpd(num_pol=5, num_mon=10, density=0.5, loop_timeout = loop_timeout)
    assert time < loop_timeout

def test_custom_log_files():
    # remove files that might've been output by other tests so that file loading
    # can fail if it's wrong
    rm_files('log.txt', 'trajectory.gsd', 'rdf.csv')

    time_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    gsd_file_name = f"{time_string}.gsd"
    log_file_name = f"{time_string}.txt"
    rdf_file_name = f"rdf.csv"

    s = dpd.create_polymer_system_dpd(
        num_pol=5,
        num_mon=10,
        density=0.5,
        gsd_file_name=gsd_file_name,
        log_file_name=log_file_name
    )

    gsd_exists = os.path.isfile(gsd_file_name)
    log_exists = os.path.isfile(log_file_name)
    rdf_exists = os.path.isfile(rdf_file_name)

    # clean up after ourselves
    rm_files(gsd_file_name, log_file_name, rdf_file_name)

    assert gsd_exists and log_exists
