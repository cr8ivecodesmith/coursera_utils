Generate Document Spec
===

## **Sources**

Main module:
- `app/generate_document.py`


## **Description**

A script that will create a new document based on given reference documents.
The document created will be using AI guided by prompts found in a config file.

## **Inputs**

**Required:**
- Document type (a string that matches a config in a `documents.toml` file)
- Output filename (in markdown)
- List of files and/or directories containing reference documents. Reference
  files can be anything that can be opened as plain text.

**Optional:**
- `--extensions`: Customize what file extensions to process. By default, only
txt and markdown files will be processed. This can be a list of values that we
will use to filter the files types to process.
- `--level-limit`: Limits the directory level to traverse when looking for
text files. Defaults to 0, for no limit. Any integer >= 1 will traverse
accordingly.
- `--config`: Path to a toml config file. Defaults to looking for a 
  `documents.toml` file where the script is run, then inside script's app
  module. Raises an error when no config is found with valid content.

**`documents.toml`**

This config contains document types to generate. The following will be provided
as defaults:

- keywords: Picks out topics or subjects from the provided files with
  A brief description for each based on how it was used or defined. Groups
  keywords found by file.
- reading\_assignment: Creates a reading article for studying from the provided
  files. Each file will be a main heading. Provide summary, key ideas, 
  questions, stories. Prefer paraphrasing directly from the files.
- book: Creates a book from the provided files. Each file becomes a chapter.
  Keeping faithful to the text is important. Reorganize into coherent
  sentences > paragraphs > topics. Decide whether to use an instructional or
  story telling voice depending on context, and keep it consistent.

Each config should have the following sections:
- model: To customize what gpt model to use.
- description: Describes what the config is for.
- prompt: Used as guidance for generating the document.
