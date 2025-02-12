#!/usr/bin/env python3

#This checks if we can recreate the viewer without a memory leak. The memory leak can happen if rust destroys the wgpu device before all the other gpu buffers
#basically checks that this bug doesn't affect Gloss
#issue: https://github.com/gfx-rs/wgpu/issues/5529
#fix: https://github.com/jonmmease/avenger/pull/67

from gloss  import *
import os
import pytest
import pytest_timeout
from gloss.log import LogLevel, gloss_setup_logger as setup_logger

# To be called only once per process. Can select between Off,Error,Warn,Info,Debug,Trace
setup_logger(log_level=LogLevel.Info) 

@pytest.mark.timeout(60, method="thread") # If it runs more than 60s, we assume it failed and got stuck
def test_viewer_recreate():

    path_root=os.path.dirname( os.path.realpath(__file__) )
    path_config=os.path.join(path_root,"./viewer_recreate.toml")

    step = 0
    while True:
        print("step",step)
        viewer = ViewerHeadless(256,256,path_config) # Resolution of the rendering window
        step+=1
        if step==500:
            break # Assume we succeeded if we can recreate X times but this is flacky