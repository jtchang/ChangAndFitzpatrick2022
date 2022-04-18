Installation
############

Setting Up Python
=================
To get started you will need to have Python 3.7 installed on your system.
Using a virtual environment and package manager (such as conda or virtualenv) is strongly suggested to manage both your python version and packages. `Anaconda`_ is an easy way to get started with conda and scientific python.

For example using conda, you would create a virtual environment by running:

.. code-block:: bash

    $ conda create -n <YOUR-ENV-NAME-HERE> python=3.7


Once your virtual environment is created, you will need to activated it.

On Windows:

.. code-block:: bash

    $ activate <YOUR-ENV-NAME-HERE>


.. note::
   Make sure that when you open a terminal or new notebook that you are running in your virtual environment.

In order to interact with your data, it is suggested that you use Jupyter Notebooks, which can be installed in conda by running:

.. code-block:: bash

    $ conda install -c anaconda jupyter


Setting Up fleappy
==================

Clone from Github
-----------------
|GIT|

.. |GIT| image:: https://imgs.xkcd.com/comics/git.png

Now that you have a python environment setup, you will need to get a copy of fleappy. It is again suggested that you use `git`_ version control to pull a local copy from the github repo. 

Once git is installed, pull down a local copy.

.. code-block:: bash

   $ git clone https://github.com/jtchang/MPFI_IC.git


If you are not comfortable using git from the command line there are a number of graphical alternatives that you may want to try including but not limited to: `Git Kraken`_, `GitHub Desktop`_, and `others`_. In addition you can manage GIT repositories in the text editors `VSCode`_ and `Atom`_. If you are new to GIT, there are numerous tutorials online, but a nice non-code heavy tutorial is `Git and Github for Poets`_.

Installation
------------

Fleappy is organized to be distributed as a PYPI package. In order to install fleappy and allow the code to be editable navigate to the fleappy directory you cloned from github in your command prompt/shell and run:

.. code-block:: bash
    
    $ pip install -e .


Configuration
-------------

Fleappy makes use of a .env file for user specific configuration. In the folder you cloned from Github, find the file named 'sample.env', make a copy of it, and rename the copy to '.env'. Two configuration settings are currently supported. The pushbullet api key, which allows for Pushbullet notifications which are useful when running large batch scripts. After signing up for an account, this api key can be obtained from `api_key`_. 

The second important field is the path to stimulus descriptions. A default file containing stimulus definitions can be found in the fleappy root directory (stim_defs.json). The absolute path to this file should be placed in the .env file (e.g. 'MYREPOSITORY/stim_defs.json').

.. Note:: 

    If you are on windows, you may run into problems using '\\' in your paths. You should be able to substitute them with '/' without any issues.

.. code-block:: python

    PUSHBULLET_API_KEY = '<YOUR API KEY HERE>'
    STIM_DEFINITIONS = 'ABSOLUTE_PATH_TO_STIM_DEFINITIONS_FILE'


.. Note::

    In Windows you may get an error message 'You must type a file name' if you try to rename the file in explorer. You make a copy and rename 'sample.env' by opening and saving as '.env' in any text editor.

Documentation
-------------

Fleappy is documented using `Sphinx`_. You can generate updated versions of the documentation by running the followin in command line.


HTML
^^^^
First we will automatically generate the documentation of the api, then we will generate the html documentation. From the fleappy home directory run:

.. code-block:: bash

    cd docs
    make html

Now that the documentation is made you can serve the path docs/_build/html with a simple http server. Python has one of these built in:

.. code-block:: bash   

    cd _build/html
    python -m http.server

Now you can reach the documentation by visiting localhost:8000 in your browser.

If you edit files within fleappy, you can run the above commands to update the documentation with any changes you have made.


PDF
^^^

Alternatively if you have miklatex installed you can generate a pdf of your documentation by running in the fleappy home directory:

.. code-block:: bash

    cd docs
    sphinx-autoapi -o ./fleappy /source
    make latex
    cd _build/latex
    pdflatex.exe fleappy.tex

You will find a fleappy.pdf in the docs/_build/latex.

If you edit files within fleappy, you can run the above commands to update the documentation with any changes you have made.

.. _api_key: https://www.pushbullet.com/#settings/account
.. _git: https://git-scm.com/
.. _Git Kraken: https://www.gitkraken.com
.. _TortoiseGit: https://tortoisegit.org/
.. _others: https://git-scm.com/downloads/guis/
.. _VSCode: https://code.visualstudio.com/docs/introvideos/versioncontrol
.. _Atom: https://flight-manual.atom.io/using-atom/sections/version-control-in-atom/
.. _GitHub Desktop: https://desktop.github.com/
.. _Git and Github for Poets: https://www.youtube.com/playlist?list=PLRqwX-V7Uu6ZF9C0YMKuns9sLDzK6zoiV
.. _Sphinx: http://www.sphinx-doc.org/en/master/
.. _Anaconda: https://www.anaconda.com/distribution
