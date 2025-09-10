# from setuptools import find_packages, setup

# setup(
#     name="medical_chatbot",
#     version="0.1.0",
#     author="Naresh",  
#     author_email="info@tconnectservices.com",  
#     packages=find_packages(),
#     install_requires=[
#         "langchain==0.3.26",
#         "flask==3.1.1",
#         "sentence-transformers==4.1.0",
#         "pypdf==5.6.1",
#         "python-dotenv==1.1.0",
#         "langchain-pinecone==0.2.8",
#         "langchain-community==0.3.26",
#         "faiss-cpu==1.7.4"
#     ],
# )

from setuptools import find_packages, setup

setup(
    name="medical_chatbot",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    package_dir={"": "src"},           # Important: tells setuptools that packages are in 'src'
    packages=find_packages(where="src"), # Find packages under 'src'
    install_requires=[
        "langchain==0.3.26",
        "flask==3.1.1",
        "sentence-transformers==4.1.0",
        "pypdf==5.6.1",
        "python-dotenv==1.1.0",
        "langchain-pinecone==0.2.8",
        "langchain-community==0.3.26",
        "faiss-cpu==1.7.4"
    ],
    python_requires=">=3.8",
)

