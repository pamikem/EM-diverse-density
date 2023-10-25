current_path=$PWD
venv_name=venv

echo "CHECK PYTHON VENV:"
if [ -d $current_path"/venv" ]
then
    echo "python venv found."
    echo "install all packages from requirements.txt ? (Y/N)"
    read varname
    if [ $varname == 'N' ]
    then
        echo "End of make env."
        exit
    else
        echo "start to install all python packages."
        source $current_path/venv/bin/activate
        pip install -r $current_path"/requirements.txt"
        pip install -r $current_path"/requirements_ds.txt"
    fi

else
    # create venv with python3.8
    echo "create the venv."
    virtualenv $current_path/$venv_name --python=3.8
    source venv/bin/activate
    echo "install all packages."
    pip install -r $current_path/requirements.txt
    # pip install -r $current_path"/requirements_ds.txt"

    echo "install venv kernel to jupyter-notebook"
    pip install ipykernel
    python -m ipykernel install --user --name=$venv_name
fi