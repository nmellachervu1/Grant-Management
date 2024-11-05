# Instructions to Setup Environment

### For Windows
`python -m venv <ENV_NAME>`

`\<ENV_NAME>\scripts\activate`

`pip install -r requirements.txt` 

- To exit environment
`deactivate`

### For Mac
`python3 -m venv <ENV_NAME>`

`source <ENV_NAME>/bin/activate`

`pip3 install -r requirements.txt` 

- To exit environment
`deactivate`

### If you need to install new packages!!
- Install new packages with pip
  - Windows: `pip install <PACKAGE_NAME>`
  - Mac: `pip3 install <PACKAGE_NAME>`
- Add packages to Requirements.txt
  - `pip freeze > requirements.txt`
 
# Instructions to Run App

Run: `flask run`

Stop: Ctrl+C
