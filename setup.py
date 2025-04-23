from setuptools import setup, find_packages

setup(
    name='hashattention',  # Replace with your package name
    version='1.0',
    packages=find_packages(include=['hashattention', 'hashattention.*']),
    include_package_data=True,
    install_requires=[],  # Add dependencies if needed
    author='Aditya Desai',
    author_email='apdesai@berkeley.edu',
    description='A simple package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/xalg-ai/hashattention1.0.git',  # Replace with your repo URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)

