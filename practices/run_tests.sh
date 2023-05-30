#!/bin/bash

# tests all the practices
cd practices

for dir in ./*; do
    if [ -d "$dir" ]; then
        echo "Testing $dir"
        cd $dir

        echo "Installing $dir"
        
        # run the install script
        chmod +x install.sh
        ./install.sh

        # check the exit code
        if [ $? -eq 1 ]; then
            echo "Install failed"
            exit 1
        fi

        # activate the virtual environment
        source venv/bin/activate

        # test the script
        python test.py

        # check the exit code
        if [ $? -eq 1 ]; then
            echo "Test failed"
            exit 1
        fi

        # deactivate the virtual environment
        deactivate

        # cleanup
        rm -rf venv
        cd ..
    fi
done