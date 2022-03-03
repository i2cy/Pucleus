::pyinstaller -F --paths=pyqtgraph.graphicsItems.ViewBox.axisCtrlTemplate_pyqt5 -n Pucleus -i icon/main_icon.ico main.py
::pyinstaller -F -p D:\Anaconda3\envs\pucleus\Lib\site-packages -n Pucleus.exe -i icon/main_icon.ico main.py
pyinstaller -F -w -n Pucleus -i icon/main_icon.ico main.py
