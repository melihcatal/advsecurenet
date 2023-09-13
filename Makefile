# remove distribution files
remove-dist:
	rm -rf dist AdvSecureNet.egg-info build

# package distribution files
package: remove-dist
	python setup.py sdist bdist_wheel

# install distribution files to create command line tool
install-cli-dev: package
	pip install -e .

test-cli-dev: 
	advsecurenet --help