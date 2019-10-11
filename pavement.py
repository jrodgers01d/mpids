#!/usr/bin/env python

from paver.easy import *
import paver.doctools
import os
import glob
import shutil

@task
def test_mpinumpy():
  sh('nosetests --exe --with-coverage --cover-erase --cover-branches --cover-package=mpids.MPInumpy mpids.MPInumpy.tests')
  pass

@task
def clean():
  for pycfile in glob.glob("*/*/*.pyc"): os.remove(pycfile)
  for pycache in glob.glob("*/__pycache__"): os.removedirs(pycache)
  for pycache in glob.glob("./__pycache__"): shutil.rmtree(pycache)
  for report in glob.glob('./*.report'):
    os.remove(report)
  try:
    shutil.rmtree(os.getcwd() + "/cover")
  except:
    pass
  pass

@task
@needs(['clean', 'test_mpinumpy'])
def default():
  pass
