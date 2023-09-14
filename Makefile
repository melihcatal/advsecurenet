# remove distribution files
remove-dist:
	rm -rf dist AdvSecureNet.egg-info build

# package distribution files
package: remove-dist
	python setup.py sdist bdist_wheel

# install distribution files to create command line tool
install-cli-dev: package
	pip install -e .

# test if command line tool is installed 
#TODO: add more comprehensive tests here
test-cli-dev: 
	advsecurenet --help