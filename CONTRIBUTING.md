# Contributing to PySAP

PySAP is a package for sparsity with applications in astrophysics and MRI.
This package has been developed in collaboration between [CosmoStat](http://www.cosmostat.org/) and [NeuroSpin](http://joliot.cea.fr/drf/joliot/Pages/Entites_de_recherche/NeuroSpin.aspx) via the [COSMIC](http://cosmic.cosmostat.org/) project.

## Contents

1. [Introduction](#introduction)  
2. [Issues](#issues)  
   a. [Asking Questions](#asking-questions)  
   b. [Installation Issues](#installation-issues)  
   c. [Reporting Bugs](#reporting-bugs)  
   d. [Requesting Features](#requesting-features)  
3. [Pull Requests](#pull-requests)  
   a. [Before Making a PR](#before-making-a-pr)  
   b. [Making a PR](#making-a-pr)  
   c. [After Making a PR](#after-making-a-pr)  
   d. [Content](#content)  
   e. [CI Tests](#ci-tests)   
   f. [Coverage](#coverage)  
   g. [Style Guide](#style-guide)  

## Introduction

PySAP is fully open-source and as such users are welcome to fork, clone and/or reuse the software freely. Users wishing to contribute to the development of this package, however, are kindly requested to adhere to the following guidelines and the [code of conduct](./CODE_OF_CONDUCT.md).

## Issues

The easiest way to contribute to PySAP is by raising a "New issue". This will give you the opportunity to ask questions, report bugs or even request new features.

Remember to use clear and descriptive titles for issues. This will help other users that encounter similar problems find quick solutions. We also ask that you read the available documentation and browse existing issues on similar topics before raising a new issue in order to avoid repetition.  

### Asking Questions

Users are of course welcome to ask any question relating to PySAP and we will endeavour to reply as soon as possible.

These issues should include the `help wanted` label.

### Installation Issues

If you encounter difficulties installing PySAP be sure to re-read the installation instructions provided. If you are still unable to install the package please remember to include the following details in the issue you raise:

* your operating system and the corresponding version (*e.g.* macOS v10.14.1, Ubuntu v16.04.1, *etc.*),
* the version of Python you are using (*e.g* v3.6.7, *etc.*),
* the python environment you are using (if any) and the corresponding version (*e.g.* virtualenv v16.1.0, conda v4.5.11, *etc.*),
* the exact steps followed while attempting to install PySAP
* and the error message printed or a screen capture of the terminal output.

These issues should include the `installation` label.

### Reporting Bugs

If you discover a bug while using PySAP please provide the same information requested for installation issues. Be sure to list the exact steps you followed that lead to the bug you encountered so that we can attempt to recreate the conditions.

If you are aware of the source of the bug we would very much appreciate if you could provide the module(s) and line number(s) affected. This will enable us to more rapidly fix the problem.

These issues should include the `bug` label.

### Requesting Features

If you believe PySAP could be improved with the addition of extra functionality or features feel free to let us know. We cannot guarantee that we will include these features, but we will certainly take your suggestions into consideration.

In order to increase your chances of having a feature included, be sure to be as clear and specific as possible as to the properties this feature should have.

These issues should include the `enhancement` label.

## Pull Requests

If you would like to take a more active roll in the development of PySAP you can do so by submitting a "Pull request". A Pull Requests (PR) is a way by which a user can submit modifications or additions to the PySAP package directly. PRs need to be reviewed by the package moderators and if accepted are merged into the master branch of the repository.

Before making a PR, be sure to carefully read the following guidelines.

### Before Making a PR

The following steps should be followed before making a pull request:

1. Log into your GitHub account or create an account if you do not already have one.

1. Go to the main PySAP repository page: [https://github.com/CEA-COSMIC/pysap](https://github.com/CEA-COSMIC/pysap)

1. Fork the repository, *i.e.* press the button on the top right with this symbol <img src="https://upload.wikimedia.org/wikipedia/commons/d/dd/Octicons-repo-forked.svg" height="20">. This will create an independent copy of the repository on your account.

1. Clone your fork of PySAP.  

```bash
  git clone https://github.com/YOUR_USERNAME/pysap
```

5. Add the original repository (*upstream*) to remote.

```bash
  git remote add upstream https://github.com/CEA-COSMIC/pysap
```

### Making a PR

The following steps should be followed to make a pull request:

1. Pull the latest updates to the original repository.

```bash
  git pull upstream master
```

2. Create a new branch for your modifications.

```bash
  git checkout -b BRANCH_NAME
```

3. Make the desired modifications to the relevant modules.

4. Add the modified files to the staging area.

```bash
  git add .
```

5. Make sure all of the appropriate files have been staged. Note that all files listed in green will be included in the following commit.

```bash
  git status
```

6. Commit the changes with an appropriate description.

```bash
  git commit -m "Description of commit"
```

7. Push the commits to a branch on your fork of PySAP.

```bash
  git push origin BRANCH_NAME
```

8. Make a pull request for your branch with a clear description of what has been done, why and what issues this relates to.

9. Wait for feedback and repeat steps 3 through 7 if necessary.

### After Making a PR

If your PR is accepted and merged it is recommended that the following steps be followed to keep your fork up to date.

1. Make sure you switch back to your local master branch.

```bash
  git checkout master
```

2. Delete the local branch you used for the PR.

```bash
  git branch -d BRANCH_NAME
```

3. Pull the latest updates to the original repository, which include your PR changes.

```bash
  git pull upstream master
```

4. Push the commits to your fork.

```bash
  git push origin master
```

### Content

Every PR should correspond to a bug fix or new feature issue that has already be raised. When you make a PR be sure to tag the issue that it resolves (*e.g.* this PR relates to issue #1). This way the issue can be closed once the PR has been merged.

The content of a given PR should be as concise as possible. To that end, aim to restrict modifications to those needed to resolve a single issue. Additional bug fixes or features should be made as separate PRs.

### CI Tests

Continuous Integration (CI) tests are implemented via [Travis CI](https://travis-ci.org/). All PRs must pass the CI tests before being merged. Your PR may not be reviewed by a moderator until all CI test are passed. Therefore, try to resolve any issues in your PR that may cause the tests to fail.

In some cases it may be necessary to modify the unit tests, but this should be clearly justified in the PR description.

### Coverage

Coverage tests are implemented via [Codecov](https://about.codecov.io/). These tests will fail if the coverage, *i.e.* the number of lines of code covered by unit tests, decreases. When submitting new code in a PR, contributors should aim to write appropriate unit tests. If the coverage drops significantly moderators may request unit tests be added before the PR is merged.

### Style Guide

All contributions should adhere to the following style guides currently implemented in PySAP:

1. All code should be compatible with the Python versions listed in `README.rst`.

1. All code should adhere to [PEP8](https://www.python.org/dev/peps/pep-0008/) standards.

1. Docstrings need to be provided for all new modules, methods and classes. These should adhere to [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) standards.

1. When in doubt look at the existing code for inspiration.
