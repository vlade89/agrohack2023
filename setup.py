from setuptools import setup

dependencies = [
    'numpy',
    'pandas',
    'scikit-learn',
    'numba',
    'pymorphy2',
    'compress_fasttext',
    'stop_words',
    'BeautifulSoup'
]

setup(
    name='agrohack2023',
    version='0.0.1',
    packages=['lib'],
    url='https://github.com/vlade89/agrohack2023',
    install_requires=dependencies,
    license='',
    author='turquoise team',
    author_email='',
    description=''
)
