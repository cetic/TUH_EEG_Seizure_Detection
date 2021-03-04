# TUH_EEG_Seizure_Detection
[![GitHub Super-Linter](https://github.com/cetic/TUH_EEG_Seizure_Detection/workflows/Lint%20Code%20Base/badge.svg)](https://github.com/marketplace/actions/super-linter)

This repository will meanly contain the Python code used Vincent STRAGIER for his intership and master dissertation at the CETIC. More information will be added to the project Wiki in order to explain how to use this pipeline.

## UMONS cluster related instructions

1 - Get access to the cluster:

```bash
ssh username@******.ac.be -p 22000
```

- username: `******************`
- password: `******************`
- password hint: `******************`

2 - [Get rid of the password](http://linuxproblem.org/art_9.html)

3 - Get [VSCode](https://code.visualstudio.com/docs/remote/ssh) working (+ use socket).

4 - [Install the Python version you want without privilege](http://thelazylog.com/install-python-as-local-user-on-linux/)

- Set it up as a module with Lmod

  - ``mkdir ~/modulefiles/python``

  - ``nano 3.8.6rc1.lua``

  - add the following lines:

    ```lua
    help([==[This module sets the PATH variable for python/3.8.6rc1]==])
    
    whatis([==[Description:Python is a programming language that lets you work quickly and integrate systems more effectively. ]==])
    whatis([==[Homepage: https://python.org ]==])
    
    local pythonroot = "$HOME/python-3.8/Python-3.8.6rc1"
    -- Workaround to use the library not compiled with Python (copied from Python 3.6)
    local lib_dynload = "$HOME/python-3.8/lib/python3.8/lib-dynload"
    -- Allows to use ipython and pip
    local python_utils = "$HOME/python-3.8/bin"
    
    conflict("python")
    --setenv("PYTHONPATH", pythonroot)
    prepend_path("PATH", pythonroot)
    prepend_path("PATH", python_utils)
    prepend_path("PYTHONPATH", pythonroot)
    
    whatis("Name         : Python 3.8.6rc1")
    whatis("Version      : 3.8.6rc1")
    whatis("Category     : Interpreter")
    whatis("Description  : Python environment ")
    whatis("URL          : https://python.org/ ")
    
    family("python")
    ```

- Load the module on startup

  - ``$nano ~/.bashrc``

  - add the following lines to the end of the file (check if python exists and if the modules folder exists):

    ```bash
    if [ -f ~/python-3.8/Python-3.8.6rc1/python ] && [ -d ~/modulefiles/python ]; then
      module use ~/modulefiles >/dev/null
      module load python/3.8.6rc1 >/dev/null
    fi
    ```

    N.B. : Remove ``>/dev/null`` if you want to show the output. Replace it with ``&> out.log`` if you want to log the output in the file ``out.log``. You can also add the line ``alias py='python'`` (Windows like command).

  - `$source ~/.bashrc` (restart the script)




 5 - Download the dataset (TUH_EEG_SZ - v1.5.2)

  ```bash
  $python ./tools/tuh_sz_download.py "https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_seizure/v1.5.2/" nedc_tuh_eeg nedc_tuh_eeg --path /home_nfs/stragierv/TUH_SZ_v1.5.2/
  ```

 6 - Remove VSCode watch on the TUH folder

- ``files.watcherExclude`` in the [VSCode settings](https://code.visualstudio.com/docs/setup/linux#_visual-studio-code-is-unable-to-watch-for-file-changes-in-this-large-workspace-error-enospc)