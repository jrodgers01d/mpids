#!/usr/bin/env python

from paver.easy import *
import paver.doctools
import os
import glob
import shutil

@task
def test_mpids_numpy_scipy():
    test_mpinumpy()
    test_mpiscipy()
    pass

@task
def test_mpids_numpy_scipy_html():
    sh("nosetests --exe --with-coverage --cover-erase --cover-branches --cover-html  --cover-package=mpids.MPInumpy --cover-package=mpids.MPIscipy ")
    pass

@task
def test_mpinumpy():
    sh('nosetests --where=./tests/MPInumpy/ --exe --with-coverage --cover-erase --cover-branches --cover-package=mpids.MPInumpy')
    pass

@task
def test_mpinumpy_html():
    sh('nosetests --where=./tests/MPInumpy/ --exe --with-coverage --cover-erase --cover-branches --cover-html --cover-package=mpids.MPInumpy')
    pass

@task
def test_mpipandas():
    sh('nosetests --where=./tests/MPIpandas/ --exe --with-coverage --cover-erase --cover-branches --cover-package=mpids.MPIpandas')
    pass

@task
def test_mpipandas_html():
    sh('nosetests --where=./tests/MPIpandas/ --exe --with-coverage --cover-erase --cover-branches --cover-html --cover-package=mpids.MPIpandas')
    pass

@task
def test_mpiscipy():
    sh('nosetests --where=./tests/MPIscipy/ --exe --with-coverage --cover-erase --cover-branches --cover-package=mpids.MPIscipy')
    pass

@task
def test_mpiscipy_html():
    sh('nosetests --where=./tests/MPIscipy/ --exe --with-coverage --cover-erase --cover-branches --cover-html --cover-package=mpids.MPIscipy')
    pass


@task
def clean():
    for pycfile in glob.glob("*/*/*/*/*/*.pyc"): os.remove(pycfile)
    for pycache in glob.glob("*/*/*/*/__pycache__"): os.removedirs(pycache)
    for pycfile in glob.glob("*/*/*/*/*.pyc"): os.remove(pycfile)
    for pycache in glob.glob("*/*/*/__pycache__"): os.removedirs(pycache)
    for pycfile in glob.glob("*/*/*/*.pyc"): os.remove(pycfile)
    for pycache in glob.glob("*/*/*/__pycache__"): os.removedirs(pycache)
    for pycfile in glob.glob("*/*/*.pyc"): os.remove(pycfile)
    for pycache in glob.glob("*/*/__pycache__"): os.removedirs(pycache)
    for pycache in glob.glob("*/__pycache__"): os.removedirs(pycache)
    for pycache in glob.glob("./__pycache__"): shutil.rmtree(pycache)
    for coverfile in glob.glob(".coverage"): os.remove(coverfile)
    for outfile in glob.glob("mpids*.out"): os.remove(outfile)
    try:
            shutil.rmtree(os.getcwd() + "/cover")
    except:
            pass
    pass

@task
@needs(['clean', 'test_mpids_numpy_scipy'])
def default():
    pass

@task
@needs(['clean', 'test_mpinumpy_html', 'test_mpiscipy_html'])
def html_coverage_report():
    pass
