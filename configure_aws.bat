@echo off
echo.
echo  Four questions follow. Type or paste each answer, then press Enter.
echo    1. AWS Access Key ID      (paste the key ID)
echo    2. AWS Secret Access Key  (paste the secret; nothing shows, that is normal)
echo    3. Default region name    (type us-east-1)
echo    4. Default output format  (type json)
echo.
"%~dp0.venv\Scripts\python.exe" -m awscli configure --profile labvision
echo.
echo  Done. You can close this window and tell Claude "done".
pause
