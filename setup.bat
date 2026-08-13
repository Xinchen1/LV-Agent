@echo off
REM Agent项目安装脚本 - Windows

echo 🚀 Setting up OpenMythos Agent...
echo ==================================

REM 检查Python
echo 1. Checking Python...
python --version || { echo ❌ Python not found & exit /b 1 }

REM 创建虚拟环境
echo 2. Creating virtual environment...
python -m venv .venv

REM 激活虚拟环境
echo 3. Activating virtual environment...
call .venv\Scripts\activate

REM 升级pip
echo 4. Upgrading pip...
python -m pip install --upgrade pip

REM 安装PyTorch
echo 5. Installing PyTorch (CPU)...
pip install torch --index-url https://download.pytorch.org/whl/cpu

REM 安装依赖
echo 6. Installing dependencies...
pip install chromadb sentence-transformers pyyaml tiktoken requests python-dotenv pydantic rich tqdm

REM 检查OpenMythos
echo 7. Checking OpenMythos...
cd ../OpenMythos-main
if pip install -e . (
    echo    ✅ OpenMythos installed
) else (
    echo    ⚠️  OpenMythos install failed - manual install required
    echo    Run: pip install -e . (in OpenMythos-main directory)
)
cd -

REM 安装当前项目
echo 8. Installing OpenMythos Agent...
pip install -e .

REM 创建数据目录
echo 9. Creating data directories...
mkdir data\experience_store 2>nul
mkdir data\strategies 2>nul
mkdir data\reflections 2>nul
mkdir data\workspace 2>nul
mkdir logs 2>nul

echo.
echo ==================================
echo ✅ Setup complete!
echo.
echo Next steps:
echo   python quick_test.py          ^> Run sanity check
echo   python -m agent_project      ^> Interactive mode
echo   python demo.py               ^> Run demo suite
echo.
echo 📝 Note:
echo   - Edit config.yaml to customize
echo   - Add .env file for API keys if needed
echo   - See README.md for full documentation
echo.
pause
