# ubuntu18.04上非root用户安装cuda9.0与cudnn



## 参考资料

1. [Linux非root用户如何优雅的安装cuda和cudnn](https://blog.csdn.net/sinat_20280061/article/details/80421532)
2. [安装cuda时 提示toolkit installation failed using unsupported compiler解决方法](https://www.cnblogs.com/cj695/p/5212848.html)



## 安装过程

1. cuda与cudnn下载

* cuda下载地址：[https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

* cudnn下载地址：[https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)(需要注册登录账号)

2. 安装cuda

* 在浏览器中找到指定版本的cuda的下载链接，如[cuda9.0](https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run).然后在命令行中下载`wget link`，如`wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run` 

* 下载完成后，使用`chmod +x filename.run`赋予执行权限，然后执行命令`./filename.run`
* 同意协议，不安装驱动，然后将cuda和cudasamples的目录修改为个人目录。安装时可能会出现错误`toolkit installation failed using unsupported compiler`，这是因为gcc版本过高引起的，解决方法为在命令后加上`--override`，即`./filename.run --override`（这种方法治标不治本，最好的解决方法是编译安装低版本的gcc）。

3. 安装cudnn

* 找到与cuda对应版本的cudnn，如[cudnn7.6.5](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/9.0_20191031/cudnn-9.0-linux-x64-v7.6.5.32.tgz)。
* 解压文件，`tar -zxvf cudnn-9.0-linux-x64-v7.6.5.32.tgz`
* 将解压出的文件夹 **include**与**lib64**中的文件复制到cuda安装位置中对应的 **include**与 **lib64**文件夹中.

4. 修改个人用户的环境变量

* 修改`～/.bashrc`文件，在文件末尾中加下以下代码（注： **/data2/ljj/software/cuda-9.0是我的cuda安装位置**）

```
export PATH=/data2/ljj/software/cuda-9.0/bin:$PATH 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data2/ljj/software/cuda-9.0/lib64/ 
```

* 执行`source ~/.baserc`使环境变量生效

