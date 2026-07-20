@echo off
REM Doble-click para levantar el servicio de preproceso.
REM Después abrí en el navegador:  http://localhost:8000/
cd /d "%~dp0"
"C:\Users\api19\AppData\Local\Programs\Python\Python312\python.exe" -m uvicorn service.app:app --host 127.0.0.1 --port 8000
pause
