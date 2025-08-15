# GitHub Copilot Instructions

This document contains guidelines and rules for GitHub Copilot when working on this project.

## General Development Rules

- **Atomic Changes**: All changes must be atomic and do exactly what was required
- **Documentation**: There is no need to create documentation and installation files unless explicitly requested
- **Language**: Only English must be used everywhere, in names, comments and descriptions

## Architecture Guidelines

- **Modular Design**: The project architecture should be modular; for each specific task, create a separate module that does exactly that
- **Data Independence**: Each module should accept a separate set of data independent of other modules and return an equally independent set of data to ensure that the module can be reused in another project
- **Thread Safety**: Each module should work with its own copy of the data so that it can be run in multithreaded mode. If this is not possible, create a large noticeable comment in the module description explaining that the module cannot work in multithreaded mode
- **Separation of Concerns**: The main program or script should, if possible, contain only the part for orchestrating modules, but not the business logic itself

## Project Management

- **Version Control**: After finishing work on some isolated feature, always offer to make a commit with the changes
- **File Organization**: Try to keep the root directory clean and not create files in it, and keep all modules in one of the subfolders
- **Dependencies**: Third party repositories should be used as submodules unless explicitly requested otherwise
- **Repository Initialization**: Any project should start with GIT and if it is not there, a repo must be created

## Code Migration

- **Safe Transfers**: When moving functionality or files or modules from one place to another, always strictly ensure that the code is transferred without changes and clearly warn about any edits that need to be made

## Project Context

This is a video flow machine learning project that processes video data and applies various transformations and analyses. The codebase includes modules for:
- Video processing and flow analysis
- Visualization components
- Machine learning models (MemFlow, VideoFlow)
- Configuration management
- Result storage and processing

When suggesting code or making changes, consider the video processing context and ensure compatibility with existing ML pipelines.
