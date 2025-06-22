import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'BSDE DSGE'
extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx_autodoc_typehints',
    'nbsphinx',
]

# Execute notebooks during the build so CI fails on execution errors
nbsphinx_execute = 'always'
source_suffix = {'.md': 'markdown'}
master_doc = 'index'
html_theme = 'alabaster'
