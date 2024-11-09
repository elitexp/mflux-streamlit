from setuptools import setup, find_packages
long_description = 'A Streamlit WebUI application for the mflux project.'
setup(
    name='mflux-streamlit',
    version='0.0.1',
    author='Sujip Maharjan',
    author_email='elitexp2008@gmail.com',
    url='https://github.com/elitexp/mflux-streamlit',
    description='A Streamlit WebUI application for the mflux project.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'mflux-streamlit=src.main:main'
        ]
    },
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
    ),
    keywords='streamlit, mflux, mlx',
    install_requires=[
        "mflux>=0.4.1,<1.0",
        "streamlit>=1.10.0,<2.0",
    ],
    zip_safe=False
)
