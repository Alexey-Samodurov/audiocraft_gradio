## How to run.
### Note: 
- All commands should be run from the root of the project
- Supported only Unix-like systems

## Run the UI server in docker container.
```
docker build -t gradio:api .
docker run -p 8083:8083 gradio:api
```