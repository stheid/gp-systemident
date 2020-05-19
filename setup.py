from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(name='gp_systemident',
      version='0.1.0',
      description='Uses GPyTorch to estimate system dynamics of an OpenAI Gym',
      long_description=readme,
      author='Stefan Heid',
      author_email='stefan.heid@upb.de',
      python_requires='>=3.8',
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Natural Language :: English',
          'Topic :: Scientific/Engineering',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.8',
      ],
      url='https://github.com/stheid/gp-systemident',
      install_requires=requirements,
      extras_require={'examples': ['openmodelica-microgrid-gym~=0.1.3']},
      license="GNU General Public License v3"
      )
