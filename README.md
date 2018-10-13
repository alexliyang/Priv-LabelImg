# Priv-LabelImg
这个是个包含五种模式，五种功能的视觉图像数据标注工具。
五个功能分别是：分类、分割、检测框、画刷以及人体关键点。

需要安装：
python3/2 (更建议在3下，2下某些功能出现问题)

pyqt5

opencv

qdarkstyle

requests

运行方式：

一、源码运行

python labelimg.py

二、windows平台下exe可执行文件运行

只需要下载Priv-LabelImg_1.0，其中有个exe文件，直接运行即可。

人体关键点功能

使用方式

使用过程截图

如何转换为coco数据集json格式，lib文件夹下有个voc_to_coco的py文件，修正其中路径即可。

如果想可视化所标注的关键点，可参考本人的另外一个工程COCOAPI_Visualition
