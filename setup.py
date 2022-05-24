import setuptools

with open('requirements.txt', 'r') as req:
    requirements = []
    for line in req:
        requirements.append(line.strip())

setuptools.setup(
    name="decavision",
    version="1.4.1",
    author="Décathlon Canada",
    author_email="sportvisionapi@decathlon.com",
    description="A package to easily train powerful image classification models using colab's free TPUs.",
    long_description="""The AI team at Décathlon Canada developed a library to help with the training of image
                        classification models. It is specially made to exploit the free TPUs that are offered
                        in Google colab notebooks. You can find the full documentation
                        [here](https://decavision-doc.herokuapp.com/)

                        ## Version 1.4.1
                        Fix hyperparameter optimization for multilabel

                        ## Version 1.4.0
                        Added multilabel classification

                        ## Version 1.3.0
                        Use suggested image sizes
                        Add EfficientNetV2 models

                        ## Version 1.2.1
                        Change to Tensorflow 2.5

                        ## Version 1.2.0
                        Add semi-supervised learning features

                        ## Version 1.1.3
                        Link to public repository

                        ## Version 1.1.2
                        Change name of package

                        ## Version 1.1.1
                        Fix typo in split_train

                        ## Version 1.1.0
                        Remove google scraping

                        ## Version 1.0.0
                        Original release""",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements,
    url='https://github.com/Decathlon/decavision.git'
)
