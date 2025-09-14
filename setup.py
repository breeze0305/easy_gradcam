from setuptools import setup, find_packages

setup(
    name="easy_gradcam",              # 套件名稱 (PyPI 上顯示的名字)
    version="0.0.5",                  # 版本號
    packages=find_packages(where="src"),   # 從 src 資料夾找套件
    package_dir={"": "src"},               # 告訴 setuptools 原始碼放在 src
    python_requires=">=3.9",
    install_requires=[                     # 依賴
        "numpy",
        "matplotlib",
        "seaborn",
        "opencv-python"
    ],
)
