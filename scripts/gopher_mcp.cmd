@echo off
setlocal
rem Do NOT use this .cmd as the MCP command. cmd.exe wrappers break stdio pipes
rem on Windows and Kilo will restart the server forever. kilo.jsonc launches
rem python.exe -u scripts\gopher_mcp.py directly.
set "ROOT=%~dp0.."
"%ROOT%\.venv\Scripts\python.exe" -u "%~dp0gopher_mcp.py" %*
