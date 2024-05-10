# Deep Learning for Visual Computing - Assignment 0

__Important: If you submit this assignment, you will receive a grade for this course at the end of the semester!__

_This text or the reference code might not be without errors. If you find a significant error (not just typos but errors that affect the assignment), please contact us via [email](mailto:dlvc@cvl.tuwien.ac.at). Students who find and report such errors will get extra points._

In this first assignment, we will set up the development environment and discuss the advantages of Deep Learning (in general).

## Part 0

All assignments will be implemented in Python 3 and [PyTorch](https://pytorch.org/). So first make sure Python 3.8 or newer is installed on your computer as that's the minimal requirement of the most recent PyTorch version. If not, [download](https://www.python.org/downloads/) and install a recent version.

Then set up, create and enable a [virtualenv](https://virtualenv.pypa.io/en/stable/). This facilitates package installation and ensures that these packages don't interfere with other Python code you might already have. Once done, make sure `$ python --version` returns something like `python 3.10.0`. Finally, install the core packages we'll need:

    pip install numpy opencv-python

The PyTorch setup varies a bit depending on the OS, see [here](https://pytorch.org/) "INSTALL PYTORCH". Use a version with CUDA only if you have an Nvidia GPU. In any case, ensure to install PyTorch version 2.0.1 or higher (latest one is 2.2.1). This is the version we will use for testing all assignments and if they fail due to version issues, you'll get significant point deductions. Confirm this via:

    python -c "import torch; print(torch.__version__)"

To confirm that all required packages haven been installed corretly, list them with `$ pip list` and take a screenshot of your terminal window for Part 1. 

## Part 1

For the second part of this assignment, we want you to think about the motivation for Deep Learning. - i.e. What are the benefits of using deep artificial neural networks?
Write a short abstract (1 page max) that should cover the following:

* What do you think are the advantages of Deep Learning compared to traditional, or classical Machine Learning methods?
* Give at least one real-world application example of Deep Learning in Visual Computing that you find interesting/impressive and explain why. You can also discuss potential negative social impacts. Provide references! 
* Include a screenshot of your package list from Part 0. 

Submit your assignment as .pdf file on [TUWEL](https://tuwel.tuwien.ac.at/course/view.php?id=63323) until __March 22 11pm__. Also, please only register for a group if you have submitted this assignment. 




