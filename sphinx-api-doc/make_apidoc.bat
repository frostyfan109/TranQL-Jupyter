@ECHO OFF
set SPHINX_APIDOC_OPTIONS=members,undoc-members
sphinx-apidoc -o ./source ../tranql_jupyter --no-toc --force
