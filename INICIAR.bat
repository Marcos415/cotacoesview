@echo off
REM Navega para a pasta do script (opcional, mas útil se o .bat for movido)
pushd "%~dp0"

echo Ativando ambiente virtual...
call venv_py310\Scripts\activate

IF %ERRORLEVEL% NEQ 0 (
    echo ERRO: Nao foi possivel ativar o ambiente virtual.
    echo Verifique se a pasta 'venv' existe e esta correta.
    pause
    exit /b %ERRORLEVEL%
)

echo Executando o script Python...
python app.py

IF %ERRORLEVEL% NEQ 0 (
    echo ERRO: Ocorreu um erro durante a execucao do script Python.
    pause
    exit /b %ERRORLEVEL%
)

echo Processo concluido.
pause
popd