# CZ Biohub publication repository template 

This repo contains a template for starting new publication repository at the Biohub. It is prepared 
to help to organize publication related materials.

## Organization
The structure of this repo is illustrated below.
```
├── figs 
├── notebooks                 
│   ├── analysis.ipynb
├── scripts                 
│   ├── analysis.sh
├── LICENSE
└── README.md
```

## Usage
To use this template as the basis of a new publication repo, follow the steps below.

1. Choose a name for your project and create a private repo for it on GitHub under the `czbiohub` organization
using the CZ Biohub publication template. By convention, project names and repo names should be the 
same, and they should be dash-separated (for example, `my-new-publication`).

2. Clone your new repo into a directory named after your new project:
```sh
git clone git@github.com/czbiohub:my-new-publication
```

3. You can start populating your publication repository with your content. You can remove the placeholder
files like `analysis.ipynb` and `analysis.sh` and place your notebooks and scripts into these folders. 
Also, you can place your figures into `figs` folder to share high-resolution version of your figures.

4. Feel free to create new folders for the files you like to share with your publication. One thing to 
note, everything on a GitHub repository is version controlled. If you like to share some datasets together
with your publication, typically a GitHub repository is not the place. We would suggest you to upload 
the datasets you like to share to a file storage service(like Google Drive, Dropbox, etc.) and provide a
link to your datasets here in your publication repository.


