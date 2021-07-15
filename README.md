# Image classification

This repository contains the code that is used to create the decavision package published on PyPI.
The package can be found on [PyPI](www.pypi.com/decavision) and is compatible with python versions 3.6 and above.

The complete documentation can be found [here](https://decavision-doc.herokuapp.com/).

## Improve the library

If you want to modify something in the code, it is as simple as cloning this repository
```
git clone https://github.com/Decathlon/decavision.git
cd decavision
```
modifying the scripts and using the code as if you were using the package itself. This can be done locally or in a 
Colab notebook if you want to exploit the free GPUs and TPUs. In that case, an example of notebook is provided in the 
documentation. If you use the repository locally, make sure that all the requirements are installed and it is 
recommended to work in a clean environment:
```
pip install -r requirements.txt
```

## Pull request

When working on the source code you have to do it in a new branch. Only the code on the master branch is considered 
for deployment so a PR must be made and accepted to push the changes online.

### pep8speaks

A [bot](https://github.com/OrkoHunter/pep8speaks) is used in the repository to analyse PRs and check if the code follows 
PEP8 conventions. The bot comments all the problems and nothing forces the changes to be done. Reviewers can use the 
comment to request for modifications. To modify the errors used by pep8speaks, take a look at the `.pep8speaks.yml` file

### PR checklist

When evaluating a pull request, here is a list of the things to check
- `setup.py` file must be updated with a new version number and explanations of the modifications in the long description
- major changes must be explained in the documentation and in that case make sure that the `index.php` and `composer.json` files exist
- make sure the code works locally and on colab with a GPU and a TPU
- make sure that all the functions have docstrings and that they are all up to date
- `conf.py` file of the docs must show the most recent version

## Deploy package

Once you have made modifications that you are satisfied with, you can push a new version of the package online. 
More info about the procedure can be found [here](https://packaging.python.org/tutorials/packaging-projects/).

You will have to update the version number in the setup.py file. Increase the first digit if you made MAJOR changes, 
the second digit for MINOR changes and the third digit for BUG FIXES.

Also make sure that you create a new release on github at the commit that you deploy. To do so you have to create a tag 
with the version number and add it to a release. The command to create a tag is
```
git tag -a vX.X.X commit_id -m "Your message here"
```
with the `commit_id``being the id of the commit that you are deploying. Then you just use
```
git push origin vX.X.X
```
to send the tag to github.

### Process (dev)

If you make modifications to the code and want to test what it will look like in the packaged form you can push your 
code to github and install the package from the repo to use it:
```
pip install git+https://github.com/Decathlon/decavision.git@branch_name
```

### Process (prod)

After testing to make sure everything works you can deploy to PyPI. To do so you have to install a few dependencies:
```
pip install setuptools wheel twine
```
You can then build the distribution by running the following line in the root directory
```
python setup.py sdist bdist_wheel
```
This will create two files (a .whl and a .tar.gz) in a 'dist' directory, with reference to the package version number
 in their name. Make sure to delete the older files if you have any. Finally you can push the package to PyPI with
```
python -m twine upload dist/*
```
This command will ask you to login to your PyPI account and you need to have the rights to the package.

## Build documentation

This project uses [sphinx](https://www.sphinx-doc.org/en/master/usage/quickstart.html) to generate the documentation. 
A large part of the docs is generated automatically from the docstrings of the functions and classes in the code. 
These are all written using the google style. Here is an example of this style:
 ```
def function(arg1, arg2):
    """
    description of the function

    Arguments:
        arg1 (type of arg1): description of arg1
        arg2 (type of arg2): description of arg2

    Returns:
        type of output: description of output
```
Using this style with sphinx is possible because of the [napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) extension.

### Process

In order to modify the documentation, the first thing to do is to clone the repository and install the requirements.
You additionally have to install sphinx and the template that we use:
```
pip install sphinx
pip install sphinx-rtd-theme
```
Then change to the 'docs' directory. This contains two types of files:
- conf.py
- .rst files

The conf.py file contains all the specifications about the project, like the extensions and their properties.
This is also where you can change the template that is used. For simple modifications, this doesn't need to be modified.

The .rst (reStructuredText) files contain the bulk of the text. The index.rst file is where the table of content is 
defined, using the following type of code:
```
.. toctree::
   :maxdepth: 3

   code
```
This takes what is written in the file code.rst and makes an element in the table of content. The rest of the files 
are built using plain reStructuredText (the sphinx doc contains a good primer on this language). To include docstrings 
from a specific script, use the following code:
```
.. automodule:: path.to.script
   :members:
```

After doing your modifications, you just have to type the following line in the command prompt, still from the 'docs' directory:
```
make html
```
The files in the folder _build will be updated with your documentation documentation. You can open 
_build/html/index.html in a browser to explore the documentation. Sometimes there may be conflicts when modifying the
pages. If you see something strange, just delete the whole content of _build and build the documentation once again.

More detail about how to use sphinx can be found in this [blog post](https://medium.com/@richdayandnight/a-simple-tutorial-on-how-to-document-your-python-project-using-sphinx-and-rinohtype-177c22a15b5b)

### Deploy documentation

For now the documentation is deployed on heroku. The deployment there is simple. Just follow the steps. This assumes
that you have the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) and that your heroku account has 
access to the decavision-doc app.

1. Make sure that you have a file called 'index.php' in the folder _build/html and that it contains the code:
   ```
   <?php include_once("index.html"); ?>
   ```
   You also need to have a file called ```composer.json``` that contains only `{}`.
2. Commit all your modifications using git
3. Login to your heroku account using the command line:
   ```
   heroku login
   ```
4. Add the heroku project as a remote for git:
   ```
   heroku git:remote -a decavision-doc
   ```
5. Push the modifications to heroku with the following line:
   ```
   git subtree push --prefix docs/_build/html heroku master
   ```
   This has to be done from the main directory and it pushes only the code in the docs/_build/html subfolder to heroku.

### To do

- use tensorflow's efficientnets
- add other pretrained models (mobilenet, B4)
- save best model when optimizing
- choose which hyperparameters to search
- use data from bucket to generate tfrecords
- use local data to generate tfrecords with TPU
- generate tfrecords in batches
- allow to unfreeze everything when training
